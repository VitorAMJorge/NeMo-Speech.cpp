// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Decoder unit tests - model-free, synthetic inputs. Covers the contracts the
// runners and endpointing depend on:
//
//   GreedyCtcDecoder   collapse rule across chunk boundaries, blank handling,
//                      last_emit_frame exactness, utterance confidence, word
//                      timings, reset_utterance vs reset semantics.
//   RnntGreedyDecoder  scripted MockRnntEngine: emission sequence, blank =
//                      frame advance, max_symbols_per_step cap, LSTM state /
//                      prev_token preserved across reset_utterance (riva EOU).
//
// Usage: ./test_decoders   (no model files, no GPU)
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "greedy_ctc_decoder.h"
#include "rnnt_greedy_decoder.h"

using namespace nemo_speech::asr;

static int g_fail = 0;
static void
check(bool ok, const char* what) {
    std::fprintf(stdout, "[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        g_fail++;
}

static bool
near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

// ---------------------------------------------------------------------------
// GreedyCtcDecoder. 4 classes: 0="▁hel", 1="lo", 2="▁world", 3=<blank>.
// Frames are log-prob rows; emit_frame() builds a row whose argmax is `id`
// with softmax probability `p` (the rest share 1-p uniformly).
// ---------------------------------------------------------------------------
namespace {

constexpr int kClasses = 4;
constexpr int kBlank = 3;

std::vector<float>
frame_for(int id, float p = 0.8f) {
    std::vector<float> row(kClasses);
    const float rest = (1.0f - p) / (kClasses - 1);
    for (int c = 0; c < kClasses; c++) row[c] = std::log(c == id ? p : rest);
    return row;
}

std::vector<float>
frames_for(const std::vector<int>& ids, float p = 0.8f) {
    std::vector<float> out;
    for (int id : ids) {
        auto row = frame_for(id, p);
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

GreedyCtcDecoder
make_greedy() {
    CtcConfig cfg;
    cfg.d_model = 8;  // unused by the decoder
    cfg.num_classes = kClasses - 1;
    cfg.blank_id = kBlank;
    return GreedyCtcDecoder(cfg, {"\xE2\x96\x81hel", "lo", "\xE2\x96\x81world"});
}

void
test_sentencepiece_punctuation_detokenization() {
    const std::vector<std::string> vocab = {
        "\xE2\x96\x81"
        "How",
        "\xE2\x96\x81old",
        "\xE2\x96\x81is",
        "\xE2\x96\x81"
        "Brooklyn",
        "\xE2\x96\x81"
        "Bridge",
        "\xE2\x96\x81?",
        "\xE2\x96\x81!",
        "\xE2\x96\x81\xE0\xA5\xA4"};
    check(
        detokenize_sentencepiece({0, 1, 2, 3, 4, 5}, vocab) == "How old is Brooklyn Bridge?",
        "sentencepiece: boundary-prefixed terminator attaches to preceding word");
    check(
        detokenize_sentencepiece({0, 6}, vocab) == "How!" &&
            detokenize_sentencepiece({0, 7}, vocab) == "How\xE0\xA5\xA4",
        "sentencepiece: ASCII and Devanagari terminators attach consistently");

    std::string incremental;
    append_sentencepiece_tokens(incremental, {0, 1}, vocab);
    append_sentencepiece_tokens(incremental, {2, 3, 4}, vocab);
    append_sentencepiece_tokens(incremental, {5}, vocab);
    check(
        incremental == detokenize_sentencepiece({0, 1, 2, 3, 4, 5}, vocab),
        "sentencepiece: incremental detokenization matches full detokenization");
}

void
test_greedy_collapse_and_blank() {
    auto dec = make_greedy();
    // [0, 0, blank, 0] -> repeat collapses, blank separates -> emits {0, 0}.
    auto fr = frames_for({0, 0, kBlank, 0});
    auto ids = dec.step(fr.data(), kClasses, 4, /*frame_offset=*/0);
    check(ids == std::vector<int>({0, 0}), "greedy: collapse rule + blank separation");
}

void
test_greedy_cross_chunk_collapse() {
    auto dec = make_greedy();
    auto a = frames_for({kBlank, 0});  // chunk 1 ends emitting id 0
    auto b = frames_for({0, kBlank});  // chunk 2 starts with the same id
    auto ids1 = dec.step(a.data(), kClasses, 2, 0);
    auto ids2 = dec.step(b.data(), kClasses, 2, 2);
    check(ids1 == std::vector<int>({0}), "greedy: chunk 1 emits once");
    check(ids2.empty(), "greedy: repeated id across chunk boundary collapses");
}

void
test_greedy_seam_dedup() {
    // Buffered-window seam: the same word straddling an emit boundary can
    // spike once per window with a blank between (each window encodes the
    // seam with slightly different context). A window-leading repeat of the
    // last committed token within kSeamMergeMaxGap(=2) frames is suppressed.
    {
        auto dec = make_greedy();
        auto a = frames_for({0, kBlank});  // window 1: spike at frame 0
        auto b = frames_for({0, kBlank});  // window 2: re-spike at frame 2
        auto ids1 = dec.step(a.data(), kClasses, 2, 0);
        auto ids2 = dec.step(b.data(), kClasses, 2, 2);
        check(ids1 == std::vector<int>({0}), "greedy: seam window 1 emits once");
        check(ids2.empty(), "greedy: seam re-spike within gap suppressed");
        // Window 3 re-spikes again: the suppressed frame extended the
        // committed token's reach, so the chain keeps deduping.
        auto ids3 = dec.step(b.data(), kClasses, 2, 4);
        check(ids3.empty(), "greedy: chained seam re-spike suppressed");
    }
    {
        // A genuine blank-separated repeat beyond the seam gap still emits.
        auto dec = make_greedy();
        auto a = frames_for({0, kBlank, kBlank, kBlank});
        auto b = frames_for({0, kBlank});
        dec.step(a.data(), kClasses, 4, 0);
        auto ids = dec.step(b.data(), kClasses, 2, 4);  // gap 4 > 2
        check(ids == std::vector<int>({0}), "greedy: distant repeat re-emits");
    }
    {
        // Mid-window blank-separated repeats are not seam artifacts and
        // still emit even within the gap.
        auto dec = make_greedy();
        auto fr = frames_for({0, kBlank, 0});
        auto ids = dec.step(fr.data(), kClasses, 3, 0);
        check(ids == std::vector<int>({0, 0}), "greedy: mid-window repeat unaffected");
    }
    {
        // reset_utterance clears the dedup state: the next utterance's first
        // token emits even as an immediate repeat.
        auto dec = make_greedy();
        auto a = frames_for({0, kBlank});
        dec.step(a.data(), kClasses, 2, 0);
        dec.reset_utterance();
        auto ids = dec.step(a.data(), kClasses, 2, 2);
        check(ids == std::vector<int>({0}), "greedy: reset_utterance clears seam state");
    }
}

void
test_greedy_last_emit_frame() {
    auto dec = make_greedy();
    check(dec.last_emit_frame() == -1, "greedy: last_emit_frame -1 before audio");
    // Non-blank at local frames 1 and 2; offset 100 -> last speech = 102.
    // The repeat at frame 2 emits no token but is still speech evidence.
    auto fr = frames_for({kBlank, 0, 0, kBlank});
    dec.step(fr.data(), kClasses, 4, /*frame_offset=*/100);
    check(dec.last_emit_frame() == 102, "greedy: last_emit_frame = last non-blank frame");
}

void
test_greedy_confidence() {
    auto dec = make_greedy();
    // Two emissions at known softmax probs 0.8 and 0.6 -> mean 0.7.
    auto f1 = frame_for(0, 0.8f);
    auto f2 = frame_for(2, 0.6f);
    std::vector<float> fr;
    fr.insert(fr.end(), f1.begin(), f1.end());
    fr.insert(fr.end(), f2.begin(), f2.end());
    dec.step(fr.data(), kClasses, 2, 0);
    check(near(dec.confidence(), 0.7f, 1e-3f), "greedy: confidence = mean token softmax");
}

void
test_greedy_word_timings() {
    auto dec = make_greedy();
    dec.set_compute_timestamps(true);
    // ▁hel(f0) lo(f1) blank ▁world(f3) -> words "hel"+"lo" [0,2) and "world" [3,4).
    auto fr = frames_for({0, 1, kBlank, 2});
    dec.step(fr.data(), kClasses, 4, 0);
    dec.finalize();  // flush the in-progress word
    const auto& w = dec.word_timings();
    bool ok = w.size() == 2 && w[0].word == "hello" && w[0].start_frame == 0 &&
              w[0].end_frame == 2 && w[1].word == "world" && w[1].start_frame == 3 &&
              w[1].end_frame == 4;
    check(ok, "greedy: word timings (boundary split, start/end frames)");
}

void
test_greedy_punctuation_word_timing() {
    CtcConfig cfg;
    cfg.d_model = 8;
    cfg.num_classes = kClasses - 1;
    cfg.blank_id = kBlank;
    GreedyCtcDecoder dec(cfg, {"\xE2\x96\x81hello", "\xE2\x96\x81?", "\xE2\x96\x81world"});
    dec.set_compute_timestamps(true);
    auto fr = frames_for({0, 1, kBlank, 2});
    dec.step(fr.data(), kClasses, 4, 0);
    dec.finalize();
    const auto& w = dec.word_timings();
    const bool ok = w.size() == 2 && w[0].word == "hello?" && w[0].start_frame == 0 &&
                    w[0].end_frame == 2 && w[1].word == "world";
    check(ok, "greedy: boundary-prefixed terminator stays with timed word");
}

void
test_greedy_reset_utterance_vs_reset() {
    auto dec = make_greedy();
    dec.set_compute_timestamps(true);
    auto fr = frames_for({0});
    dec.step(fr.data(), kClasses, 1, 0);
    dec.finalize();
    dec.reset_utterance();
    check(dec.word_timings().empty(), "greedy: reset_utterance clears word timings");
    check(near(dec.confidence(), 1.0f), "greedy: reset_utterance clears confidence");
    check(dec.last_emit_frame() == 0, "greedy: reset_utterance keeps last_emit_frame");
    // reset_utterance clears the collapse state (utterance boundary): the same
    // id right after the EOU is a new utterance's first token and must emit.
    auto ids = dec.step(fr.data(), kClasses, 1, 1);
    check(ids == std::vector<int>({0}), "greedy: reset_utterance clears prev id (re-emits)");
    // Full reset also clears collapse state.
    dec.reset();
    check(dec.last_emit_frame() == -1, "greedy: reset clears last_emit_frame");
    ids = dec.step(fr.data(), kClasses, 1, 0);
    check(ids == std::vector<int>({0}), "greedy: reset clears prev id (re-emits)");
}

void
test_greedy_open_word_needs_finalize() {
    auto dec = make_greedy();
    dec.set_compute_timestamps(true);
    // "▁hel" opens a word with no closing boundary piece - it stays open.
    auto fr = frames_for({0});
    dec.step(fr.data(), kClasses, 1, 0);
    check(dec.word_timings().empty(), "greedy: open trailing word absent before finalize");
    dec.finalize();
    const auto& w = dec.word_timings();
    check(
        w.size() == 1 && w[0].word == "hel",
        "greedy: finalize flushes the open trailing word into word_timings");
}

// ---------------------------------------------------------------------------
// RnntGreedyDecoder with a scripted staged engine. Each joint round supplies
// argmax ids for all remaining frames under one predictor state. Predictor h/c
// candidates are stamped with a per-call serial so blank caching and state
// threading are observable.
// ---------------------------------------------------------------------------
class MockRnntEngine : public RnntEngine {
   public:
    MockRnntEngine() {
        cfg_.d_model = 4;
        cfg_.vocab_size = 5;  // ids 0..3 + blank
        cfg_.blank_id = 4;
        cfg_.pred_hidden = 3;
        cfg_.pred_num_layers = 1;
        cfg_.joint_dim = 4;
        cfg_.max_symbols_per_step = 3;
        vocab_ = {
            "\xE2\x96\x81"
            "a",
            "\xE2\x96\x81"
            "b",
            "\xE2\x96\x81"
            "c",
            "\xE2\x96\x81?"};
    }

    const RnntConfig& rnnt_config() const override { return cfg_; }
    const std::vector<std::string>& vocab() const override { return vocab_; }
    void set_piece(int id, const std::string& text) { vocab_.at(id) = text; }

    struct State : RnntStreamState {
        float bank[2] = {0.0f, 0.0f};
    };

    std::unique_ptr<RnntStreamState> make_rnnt_stream_state() override {
        return std::make_unique<State>();
    }

    void predict_rnnt(RnntStreamState& opaque, int prev_token, int active_bank) override {
        auto& state = static_cast<State&>(opaque);
        last_prev_token = prev_token;
        last_h_in = state.bank[active_bank];
        predictor_calls++;
        serial += 1.0f;
        state.bank[active_bank ^ 1] = serial;
    }

    void joint_argmax(
        RnntStreamState& /*state*/, const float* /*enc_proj*/, int joint_dim, int T,
        int32_t* token_ids, const float* logit_bias = nullptr) override {
        check(joint_dim == cfg_.joint_dim, "rnnt mock: projected width");
        joint_calls++;
        call_frames.push_back(T);
        call_blank_bias.push_back(logit_bias != nullptr ? logit_bias[cfg_.blank_id] : 0.0f);
        if (logit_bias != nullptr) {
            last_logit_bias.assign(logit_bias, logit_bias + cfg_.vocab_size);
        } else {
            last_logit_bias.clear();
        }
        std::vector<int> round;
        if (!rounds.empty()) {
            round = std::move(rounds.front());
            rounds.pop_front();
        }
        for (int i = 0; i < T; ++i) {
            token_ids[i] = (i < static_cast<int>(round.size())) ? round[i] : cfg_.blank_id;
        }
    }

    void joint_tdt_argmax(RnntStreamState&, const float*, int, int, int32_t*, int32_t*) override {
        throw std::runtime_error("RNNT mock does not implement TDT");
    }

    std::deque<std::vector<int>> rounds;
    int last_prev_token = -100;
    float last_h_in = -1.0f;
    float serial = 0.0f;
    int predictor_calls = 0;
    int joint_calls = 0;
    std::vector<float> last_logit_bias;
    std::vector<int> call_frames;        // T of each joint call
    std::vector<float> call_blank_bias;  // bias at blank of each joint call (0 = none)

   private:
    RnntConfig cfg_;
    std::vector<std::string> vocab_;
};

void
test_rnnt_emission_and_blank() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    // T=3 frames. f0 emits 0 then blank; f1 blank; f2 emits 1, 2, blank.
    // Each vector is one speculative joint pass over the remaining frames.
    eng.rounds = {{0, blank, blank}, {blank, blank, 1}, {2}, {blank}};
    std::vector<float> enc(3 * 4, 0.0f);
    auto ids = dec.step(enc.data(), 4, 3, /*frame_offset=*/10);
    check(ids == std::vector<int>({0, 1, 2}), "rnnt: scripted emission sequence");
    check(dec.last_emit_frame() == 12, "rnnt: last_emit_frame = last emitting frame");
    check(eng.predictor_calls == 4, "rnnt: predictor runs only initially and after emits");
    check(eng.joint_calls == 4, "rnnt: joint consumes blank runs in vectorized rounds");
    const auto& stats = dec.stats();
    check(
        stats.encoder_frames == 3 && stats.emitted_tokens == 3 && stats.predictor_calls == 4 &&
            stats.joint_calls == 4 && stats.joint_frames == 8,
        "rnnt: staged execution counters");
}

void
test_rnnt_max_symbols_cap() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    // Script never returns blank: the inner loop must stop at
    // max_symbols_per_step (3) emissions for the single frame.
    eng.rounds = {{0}, {1}, {2}, {3}, {0}, {1}};
    std::vector<float> enc(4, 0.0f);
    auto ids = dec.step(enc.data(), 4, 1, 0);
    check(static_cast<int>(ids.size()) == 3, "rnnt: max_symbols_per_step caps the inner loop");
}

void
test_rnnt_finalization_preserves_native_punctuation() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);

    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 0);
    check(eng.last_logit_bias.empty(), "rnnt: initial silence has no punctuation bias");

    eng.rounds = {{0}, {blank}};
    dec.step(enc.data(), 4, 1, 1);
    eng.rounds = {{blank}, {3}};
    const auto tail = dec.step(enc.data(), 4, 1, 2);
    const int joints = eng.joint_calls;
    dec.finalize();
    check(
        tail.empty() && eng.last_logit_bias.empty() && eng.joint_calls == joints &&
            dec.stats().emitted_tokens == 1,
        "rnnt: a clipped stream closes without forcing terminal punctuation");

    dec.reset();
    dec.set_compute_timestamps(true);
    eng.rounds = {{0}, {blank}};
    dec.step(enc.data(), 4, 1, 0);
    eng.rounds = {{3}, {blank}};
    const auto native = dec.step(enc.data(), 4, 1, 1);
    dec.finalize();
    check(
        native == std::vector<int>{3} && eng.last_logit_bias.empty() &&
            dec.word_timings().size() == 1 && dec.word_timings()[0].word == "a?",
        "rnnt: finalization preserves native terminal punctuation");
}

void
test_rnnt_stalls_do_not_inject_punctuation() {
    MockRnntEngine eng;
    eng.set_piece(2, ",");
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);
    for (bool formatting : {false, true}) {
        dec.reset();
        AsrRequestOptions opts;
        opts.enable_automatic_punctuation = formatting;
        opts.enable_word_time_offsets = true;
        dec.set_request_options(opts);
        eng.rounds = {{0}, {blank}};
        dec.step(enc.data(), 4, 1, 0);
        const int predictors = eng.predictor_calls;
        for (int frame = 1; frame <= 100; ++frame) {
            const int joints = eng.joint_calls;
            // A second query would return punctuation: it must never be made.
            eng.rounds = {{blank}, {3}};
            const auto tokens = dec.step(enc.data(), 4, 1, frame);
            check(
                tokens.empty() && eng.last_logit_bias.empty() && eng.joint_calls == joints + 1 &&
                    eng.predictor_calls == predictors && dec.stats().emitted_tokens == 1 &&
                    dec.last_emit_frame() == 0 && dec.last_speech_frame() == 0,
                "rnnt: prolonged stalls neither bias punctuation nor run a suggestion query");
        }
        eng.rounds = {{3}, {blank}};
        const auto native = dec.step(enc.data(), 4, 1, 101);
        dec.finalize();
        check(
            native == std::vector<int>{3} && eng.last_logit_bias.empty() &&
                dec.word_timings().size() == 1 && dec.word_timings()[0].word == "a?" &&
                dec.word_timings()[0].end_frame == 1,
            "rnnt: native punctuation survives prolonged stalls without changing word duration");
        dec.reset_utterance();
        for (int frame = 102; frame <= 110; ++frame) {
            eng.rounds = {{blank}, {3}};
            check(
                dec.step(enc.data(), 4, 1, frame).empty() && eng.last_logit_bias.empty(),
                "rnnt: soft EOU does not enable punctuation bias");
        }
        eng.rounds = {{blank}};
        dec.step(enc.data(), 4, 1, 111);
        dec.finalize();
        check(
            eng.last_logit_bias.empty() && dec.word_timings().empty(),
            "rnnt: closing after an endpoint cannot synthesize another punctuation final");
    }
}

void
test_rnnt_late_punctuation_returns_to_previous_final() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);
    auto step = [&](std::vector<int> round, int64_t frame) {
        eng.rounds = {std::move(round), {blank}};
        return dec.step(enc.data(), 4, 1, frame);
    };

    step({3}, 0);
    check(
        dec.take_late_punctuation().empty(),
        "rnnt: leading punctuation at stream start is dropped");

    step({0}, 1);
    dec.reset_utterance();
    check(step({3}, 2).empty(), "rnnt: late punctuation is not part of the next utterance");
    check(dec.take_late_punctuation().empty(), "rnnt: late punctuation waits for the next word");
    eng.set_piece(
        1,
        "\xE2\x96\x81"
        "B");
    step({1}, 3);
    check(
        dec.take_late_punctuation() == "?",
        "rnnt: late punctuation before a capitalized word returns to the previous final");
    check(dec.take_late_punctuation().empty(), "rnnt: late punctuation is delivered once");

    dec.reset_utterance();
    step({3}, 4);
    step({2}, 5);
    check(
        dec.take_late_punctuation().empty(),
        "rnnt: a late sentence mark before a lowercase word is dropped");

    step({3}, 6);
    step({0}, 7);
    check(dec.take_late_punctuation().empty(), "rnnt: punctuation inside an utterance is not late");

    dec.reset_utterance();
    step({3}, 8);
    check(
        dec.take_late_punctuation(/*end_of_stream=*/true) == "?",
        "rnnt: end of stream releases a waiting late mark");

    step({0}, 9);
    dec.reset_utterance();
    step({3}, 10);
    dec.reset();
    check(dec.take_late_punctuation().empty(), "rnnt: hard reset discards late punctuation");
}

void
test_rnnt_endpoint_punctuation_continuity() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    AsrRequestOptions opts;
    opts.enable_word_time_offsets = true;
    opts.enable_automatic_punctuation = true;
    dec.set_request_options(opts);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);
    eng.rounds = {{0}, {blank}};
    dec.step(enc.data(), 4, 1, 0);
    for (int frame = 1; frame <= 8; ++frame) {
        eng.rounds = {{blank}};
        dec.step(enc.data(), 4, 1, frame);
    }
    check(eng.last_logit_bias.empty(), "rnnt: continuing silence never enables punctuation bias");
    eng.rounds = {{3}, {blank}};
    dec.step(enc.data(), 4, 1, 9);
    dec.finalize();
    check(
        dec.word_timings().size() == 1 && dec.word_timings()[0].word == "a?" &&
            dec.word_timings()[0].end_frame == 1 && dec.last_speech_frame() == 0,
        "rnnt: terminal punctuation stays in the published word");
    const int predictor_calls = eng.predictor_calls;
    dec.reset_utterance();
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 10);
    check(
        eng.predictor_calls == predictor_calls && eng.last_logit_bias.empty(),
        "rnnt: soft EOU preserves cached predictor without injecting punctuation bias");
    eng.rounds = {{3}, {blank}};
    auto late = dec.step(enc.data(), 4, 1, 11);
    dec.finalize();
    check(
        late.empty() && dec.word_timings().empty() && eng.last_prev_token == 3,
        "rnnt: late punctuation updates predictor but creates no next-final text or word");
    check(
        dec.last_emit_frame() == 11 && dec.last_speech_frame() == 0,
        "rnnt: late punctuation cannot re-arm the endpoint speech clock");
    eng.rounds = {{1}, {blank}};
    auto next = dec.step(enc.data(), 4, 1, 12);
    eng.rounds = {{3}, {blank}};
    dec.step(enc.data(), 4, 1, 13);
    eng.rounds = {{3}, {blank}};
    auto repeated = dec.step(enc.data(), 4, 1, 14);
    dec.finalize();
    check(
        next == std::vector<int>{1} && repeated.empty() && dec.word_timings().size() == 1 &&
            dec.word_timings()[0].word == "b?" && dec.word_timings()[0].start_frame == 12 &&
            dec.word_timings()[0].end_frame == 13,
        "rnnt: next utterance retains text, punctuation and absolute timings without duplicates");
    dec.reset();
    check(dec.last_speech_frame() == -1, "rnnt: hard reset clears endpoint speech evidence");
    for (int frame = 0; frame < 8; ++frame) {
        eng.rounds = {{blank}};
        dec.step(enc.data(), 4, 1, frame);
    }
    check(eng.last_logit_bias.empty(), "rnnt: initial silence cannot accumulate punctuation bias");
    check(
        !sp_is_punctuation("-") && !sp_is_punctuation("'") && !sp_is_punctuation("\"") &&
            !sp_is_punctuation("don't") && !sp_is_punctuation("-5"),
        "rnnt: publication cleanup preserves signs, quotes and contractions");
    eng.set_piece(2, "\xE2\x96\x81");
    dec.reset();
    eng.rounds = {{0}, {blank}};
    dec.step(enc.data(), 4, 1, 0);
    eng.rounds = {{2}, {blank}};
    dec.step(enc.data(), 4, 1, 1);
    eng.rounds = {{3}, {blank}};
    dec.step(enc.data(), 4, 1, 2);
    eng.rounds = {{1}, {blank}};
    dec.step(enc.data(), 4, 1, 3);
    dec.finalize();
    check(
        dec.word_timings().size() == 2 && dec.word_timings()[0].word == "a?" &&
            dec.word_timings()[1].word == "b",
        "rnnt: bare word separator cannot detach punctuation into its own timed word");
}

void
test_rnnt_state_threading_and_reset_utterance() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> enc(4, 0.0f);

    // Frame emits token 2 (state serial 1 latched), then blank (serial 2
    // not latched: blank leaves the LSTM state untouched).
    eng.rounds = {{2}, {blank}};
    dec.step(enc.data(), 4, 1, 0);
    check(eng.predictor_calls == 2, "rnnt: predictor candidate cached after blank");

    // Next call: the decoder must present the post-emit state (serial 1)
    // and prev_token=2.
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 1);
    check(eng.last_prev_token == 2, "rnnt: prev_token carried across steps");
    check(near(eng.last_h_in, 1.0f), "rnnt: LSTM state latched on emit only (not on blank)");
    check(eng.predictor_calls == 2, "rnnt: blank cache survives step boundary");

    // riva EOU semantics: reset_utterance keeps predictor state + prev_token.
    dec.reset_utterance();
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 2);
    check(eng.last_prev_token == 2, "rnnt: prev_token survives reset_utterance");
    check(near(eng.last_h_in, 1.0f), "rnnt: LSTM state survives reset_utterance");
    check(dec.last_emit_frame() == 0, "rnnt: last_emit_frame survives reset_utterance");
    check(eng.predictor_calls == 2, "rnnt: blank cache survives reset_utterance");

    // Full reset: prev_token back to blank, state zeroed.
    dec.reset();
    eng.rounds = {{blank}};
    dec.step(enc.data(), 4, 1, 0);
    check(eng.last_prev_token == blank, "rnnt: reset restores prev_token = blank");
    check(near(eng.last_h_in, 0.0f), "rnnt: reset zeroes the LSTM state");
    check(dec.last_emit_frame() == -1, "rnnt: reset clears last_emit_frame");
}

void
test_rnnt_word_timings() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    const int blank = eng.rnnt_config().blank_id;
    // f0 emits "▁a", f1 emits "▁b" -> two words at global frames 5 and 6.
    eng.rounds = {{0, blank}, {blank, 1}, {blank}};
    std::vector<float> enc(2 * 4, 0.0f);
    dec.step(enc.data(), 4, 2, /*frame_offset=*/5);
    dec.finalize();
    const auto& w = dec.word_timings();
    bool ok = w.size() == 2 && w[0].word == "a" && w[0].start_frame == 5 && w[1].word == "b" &&
              w[1].start_frame == 6;
    check(ok, "rnnt: word timings on scripted emissions");
}

void
test_rnnt_punctuation_word_timing() {
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    const int blank = eng.rnnt_config().blank_id;
    // f0 emits "▁a", f1 emits "▁?": punctuation attaches without extending speech time.
    eng.rounds = {{0, blank}, {blank, 3}, {blank}};
    std::vector<float> enc(2 * 4, 0.0f);
    dec.step(enc.data(), 4, 2, /*frame_offset=*/5);
    dec.finalize();
    const auto& w = dec.word_timings();
    const bool ok =
        w.size() == 1 && w[0].word == "a?" && w[0].start_frame == 5 && w[0].end_frame == 6;
    check(ok, "rnnt: boundary-prefixed terminator stays with timed word");
}

void
test_rnnt_open_word_needs_finalize() {
    // The contract CacheStreamRunner::finalize() relies on: an open trailing
    // word reaches word_timings() only after finalize(). Without the runner
    // calling head_->finalize() at EOS, the last word's WordInfo is dropped.
    MockRnntEngine eng;
    RnntGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    eng.rounds = {{0}, {eng.rnnt_config().blank_id}};  // emit "▁a", then blank
    std::vector<float> enc(eng.rnnt_config().d_model, 0.0f);
    dec.step(enc.data(), eng.rnnt_config().d_model, 1, /*frame_offset=*/0);
    check(dec.word_timings().empty(), "rnnt: open trailing word absent before finalize");
    dec.finalize();
    const auto& w = dec.word_timings();
    check(
        w.size() == 1 && w[0].word == "a",
        "rnnt: finalize flushes the open trailing word into word_timings");
}

class MockTdtEngine : public RnntEngine {
   public:
    MockTdtEngine() {
        cfg_.d_model = cfg_.joint_dim = 2;
        cfg_.vocab_size = 4;
        cfg_.blank_id = 3;
        cfg_.max_symbols_per_step = 3;
        cfg_.durations = {0, 1, 2};
        vocab_ = {
            "\xE2\x96\x81"
            "a",
            "\xE2\x96\x81"
            "b",
            "c"};
    }
    const RnntConfig& rnnt_config() const override { return cfg_; }
    const std::vector<std::string>& vocab() const override { return vocab_; }
    struct State : RnntStreamState {};
    std::unique_ptr<RnntStreamState> make_rnnt_stream_state() override {
        return std::make_unique<State>();
    }
    void predict_rnnt(RnntStreamState&, int prev_token, int) override {
        last_prev_token = prev_token;
        ++predictor_calls;
    }
    void joint_argmax(RnntStreamState&, const float*, int, int, int32_t*, const float*) override {
        throw std::runtime_error("TDT mock does not implement RNNT argmax");
    }
    void joint_tdt_argmax(
        RnntStreamState&, const float*, int joint_dim, int T, int32_t* tokens,
        int32_t* durations) override {
        check(joint_dim == cfg_.joint_dim && T == 1, "tdt: scalar joint shape");
        ++joint_calls;
        const auto next = rounds.empty() ? std::pair<int, int>{cfg_.blank_id, 1} : rounds.front();
        if (!rounds.empty())
            rounds.pop_front();
        tokens[0] = next.first;
        durations[0] = next.second;
    }

    std::deque<std::pair<int, int>> rounds;
    int predictor_calls = 0;
    int joint_calls = 0;
    int last_prev_token = -1;

   private:
    RnntConfig cfg_;
    std::vector<std::string> vocab_;
};

void
test_tdt_duration_and_state_commit() {
    MockTdtEngine eng;
    TdtGreedyDecoder dec(&eng);
    dec.set_compute_timestamps(true);
    const int blank = eng.rnnt_config().blank_id;
    // t=0 emits a with duration 0, then blank advances to t=1. A duration-2
    // blank skips to t=3, where b emits with duration 2 and finishes T=5.
    eng.rounds = {{0, 0}, {blank, 1}, {blank, 2}, {1, 2}};
    std::vector<float> enc(5 * 2, 0.0f);
    const auto ids = dec.step(enc.data(), 2, 5, 10);
    dec.finalize();
    check(ids == std::vector<int>({0, 1}), "tdt: token and duration sequence");
    check(eng.predictor_calls == 2, "tdt: predictor commits only on non-blank");
    check(dec.last_emit_frame() == 13, "tdt: duration advances encoder frame");
    const auto& words = dec.word_timings();
    check(
        words.size() == 2 && words[0].start_frame == 10 && words[0].end_frame == 11 &&
            words[1].start_frame == 13 && words[1].end_frame == 15,
        "tdt: token timestamps include predicted span");
}

void
test_tdt_zero_duration_guard() {
    MockTdtEngine eng;
    TdtGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    eng.rounds = {{blank, 0}};
    std::vector<float> enc(2, 0.0f);
    const auto ids = dec.step(enc.data(), 2, 1, 0);
    check(ids.empty() && eng.joint_calls == 1, "tdt: blank duration zero advances immediately");
}

void
test_tdt_duration_crosses_step_boundary() {
    MockTdtEngine eng;
    TdtGreedyDecoder dec(&eng);
    const int blank = eng.rnnt_config().blank_id;
    std::vector<float> frame(2, 0.0f);
    eng.rounds = {{blank, 2}};
    check(dec.step(frame.data(), 2, 1, 0).empty(), "tdt: boundary setup is blank");

    const int calls_before_skip = eng.joint_calls;
    check(dec.step(frame.data(), 2, 1, 1).empty(), "tdt: carried duration skips a whole block");
    check(eng.joint_calls == calls_before_skip, "tdt: skipped block performs no joint call");

    eng.rounds = {{0, 1}};
    const auto ids = dec.step(frame.data(), 2, 1, 2);
    check(ids == std::vector<int>({0}), "tdt: emits after carried duration");
    check(dec.last_emit_frame() == 2, "tdt: carries duration across step boundaries");
}

// Arbitrary; exact in binary and unlike any other constant in these tests.
constexpr float kBlankPenalty = 2.25f;

DecoderConfig
penalty_config(int after_frames) {
    DecoderConfig cfg;
    cfg.blank_penalty = kBlankPenalty;
    cfg.blank_penalty_after_frames = after_frames;
    return cfg;
}

// MockRnntEngine ignores encoder values; only the frame count matters.
std::vector<int>
step_frames(RnntGreedyDecoder& dec, const MockRnntEngine& eng, int frames, int64_t frame_offset) {
    const int joint_dim = eng.rnnt_config().joint_dim;
    std::vector<float> enc(static_cast<size_t>(frames) * joint_dim, 0.0f);
    return dec.step(enc.data(), joint_dim, frames, frame_offset);
}

void
test_rnnt_blank_penalty_after_silence() {
    const float p = kBlankPenalty;
    {
        // Off by default: no bias on any call.
        MockRnntEngine eng;
        RnntGreedyDecoder dec(&eng);
        const int blank = eng.rnnt_config().blank_id;
        eng.rounds = {{blank, blank, blank}};
        step_frames(dec, eng, /*frames=*/3, /*frame_offset=*/0);
        check(eng.last_logit_bias.empty(), "blank penalty: off by default");
        check(eng.call_frames == std::vector<int>({3}), "blank penalty: off never splits a run");
    }
    {
        // N = 2, one chunk of frames 0-4.
        MockRnntEngine eng;
        RnntGreedyDecoder dec(&eng, penalty_config(/*after_frames=*/2));
        const int blank = eng.rnnt_config().blank_id;
        // round 1, frames 0-4: nothing emitted yet, penalty on; emits 0 at frame 0
        // round 2, frames 0-1: reset by that token, penalty off; all blank
        // round 3, frames 2-4: 2 frames since the token, penalty on; emits 1 at frame 3
        // round 4, frames 3-4: reset again (3 + 2 = 5 is past the chunk); all blank
        eng.rounds = {{0}, {blank, blank}, {blank, 1}, {blank, blank}};
        const auto ids = step_frames(dec, eng, /*frames=*/5, /*frame_offset=*/0);
        check(ids == std::vector<int>({0, 1}), "blank penalty: emission sequence");
        check(
            eng.call_frames == std::vector<int>({5, 2, 3, 2}),
            "blank penalty: the run splits at frame last + N");
        check(
            near(eng.call_blank_bias[0], -p) && near(eng.call_blank_bias[1], 0.0f) &&
                near(eng.call_blank_bias[2], -p) && near(eng.call_blank_bias[3], 0.0f),
            "blank penalty: on from the start and after N silent frames, off after any token");
        check(dec.last_emit_frame() == 3, "blank penalty: the last token restarts the count");
        // The count carries across chunks: frames 5-6 of the next chunk are past 3 + 2.
        eng.rounds = {{blank, blank}};
        step_frames(dec, eng, /*frames=*/2, /*frame_offset=*/5);
        check(
            near(eng.call_blank_bias.back(), -p), "blank penalty: the count carries across chunks");
    }
    {
        // N = 0: a global penalty, every call biased.
        MockRnntEngine eng;
        RnntGreedyDecoder dec(&eng, penalty_config(/*after_frames=*/0));
        const int blank = eng.rnnt_config().blank_id;
        eng.rounds = {{0}, {blank, 1}, {blank}};
        step_frames(dec, eng, /*frames=*/2, /*frame_offset=*/0);
        bool all = !eng.call_blank_bias.empty();
        for (float b : eng.call_blank_bias) all = all && near(b, -p);
        check(all, "blank penalty: after_frames 0 applies it on every frame");
    }
    {
        // Only blank is biased: token and punctuation logits stay native.
        MockRnntEngine eng;
        RnntGreedyDecoder dec(&eng, penalty_config(/*after_frames=*/0));
        const int blank = eng.rnnt_config().blank_id;
        eng.rounds = {{blank}};
        step_frames(dec, eng, /*frames=*/1, /*frame_offset=*/0);
        bool others_zero = eng.last_logit_bias.size() == static_cast<size_t>(blank) + 1;
        for (int id = 0; others_zero && id < blank; ++id)
            others_zero = near(eng.last_logit_bias[static_cast<size_t>(id)], 0.0f);
        check(
            others_zero && near(eng.last_logit_bias[static_cast<size_t>(blank)], -p),
            "blank penalty: biases blank only");
    }
    {
        // The max_symbols cap advances the clock: with N = 1 the frame after the
        // capped one is one frame past the last token, so it is penalized.
        MockRnntEngine eng;  // max_symbols_per_step = 3
        RnntGreedyDecoder dec(&eng, penalty_config(/*after_frames=*/1));
        const int blank = eng.rnnt_config().blank_id;
        eng.rounds = {{0}, {1}, {2}, {blank}};
        const auto ids = step_frames(dec, eng, /*frames=*/2, /*frame_offset=*/0);
        check(ids == std::vector<int>({0, 1, 2}), "blank penalty: the symbol cap still applies");
        check(
            eng.call_frames.back() == 1 && near(eng.call_blank_bias.back(), -p),
            "blank penalty: after the cap, the next frame is N = 1 past the last token");
    }
    {
        // reset() starts a new stream: penalized again from its first frame.
        // reset_utterance() (an end of utterance) keeps the count running.
        MockRnntEngine eng;
        RnntGreedyDecoder dec(&eng, penalty_config(/*after_frames=*/4));
        const int blank = eng.rnnt_config().blank_id;
        eng.rounds = {{0}, {blank}};
        step_frames(dec, eng, /*frames=*/1, /*frame_offset=*/0);
        check(near(eng.call_blank_bias.back(), 0.0f), "blank penalty: off right after a token");
        dec.reset_utterance();
        eng.rounds = {{blank}};
        step_frames(dec, eng, /*frames=*/1, /*frame_offset=*/1);
        check(
            near(eng.call_blank_bias.back(), 0.0f),
            "blank penalty: reset_utterance() keeps the count");
        dec.reset();
        eng.rounds = {{blank}};
        step_frames(dec, eng, /*frames=*/1, /*frame_offset=*/2);
        check(near(eng.call_blank_bias.back(), -p), "blank penalty: reset() turns it back on");
    }
    const std::vector<std::pair<double, int>> invalid = {
        {-1.0, 0},
        {1.0, -1},
        {std::numeric_limits<double>::quiet_NaN(), 0},
        {std::numeric_limits<double>::infinity(), 0}};
    for (const auto& bad : invalid) {
        DecoderConfig cfg;
        cfg.blank_penalty = bad.first;
        cfg.blank_penalty_after_frames = bad.second;
        MockRnntEngine eng;
        bool threw = false;
        try {
            RnntGreedyDecoder dec(&eng, cfg);
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "blank penalty: negative or non-finite values are refused");
    }
}

// ---------------------------------------------------------------------------
// Blank penalty under concurrency, checked against the rule itself rather than
// against a transcript. Each stream has a script (absolute frame -> token); the
// engine emits a scripted token once, at its frame, and blank everywhere else.
// Encoder values carry the absolute frame, so the engine needs no call order.
// Every joint call is recorded on the stream's own state; the oracle replays
// them and checks the bias each one carried.
// ---------------------------------------------------------------------------
class ScriptedRnntEngine : public RnntEngine {
   public:
    struct Call {
        int64_t first_frame;
        int frames;
        float blank_bias;
    };
    struct State : RnntStreamState {
        std::map<int64_t, int> pending;  // frame -> token not yet emitted
        std::vector<Call> calls;
    };

    explicit ScriptedRnntEngine(std::vector<std::map<int64_t, int>> scripts)
        : scripts_(std::move(scripts)) {
        cfg_.d_model = 2;
        cfg_.vocab_size = 4;  // ids 0..2 + blank
        cfg_.blank_id = 3;
        cfg_.pred_hidden = 2;
        cfg_.pred_num_layers = 1;
        cfg_.joint_dim = 2;
        cfg_.max_symbols_per_step = 3;
        vocab_ = {
            "\xE2\x96\x81"
            "a",
            "\xE2\x96\x81"
            "b",
            "\xE2\x96\x81"
            "c"};
    }

    const RnntConfig& rnnt_config() const override { return cfg_; }
    const std::vector<std::string>& vocab() const override { return vocab_; }

    // Streams are created one per decoder, in construction order.
    std::unique_ptr<RnntStreamState> make_rnnt_stream_state() override {
        std::lock_guard<std::mutex> lock(mu_);
        auto state = std::make_unique<State>();
        state->pending = scripts_.at(states_.size() % scripts_.size());
        states_.push_back(state.get());
        return state;
    }

    void predict_rnnt(RnntStreamState&, int, int) override {}

    // Touches only the caller's own state: safe from concurrent streams.
    void joint_argmax(
        RnntStreamState& opaque, const float* enc_proj, int joint_dim, int T, int32_t* token_ids,
        const float* logit_bias = nullptr) override {
        auto& state = static_cast<State&>(opaque);
        const auto frame = [&](int i) {
            return static_cast<int64_t>(enc_proj[static_cast<size_t>(i) * joint_dim]);
        };
        state.calls.push_back({frame(0), T, logit_bias ? logit_bias[cfg_.blank_id] : 0.0f});
        bool committed = false;
        for (int i = 0; i < T; ++i) {
            const auto it = state.pending.find(frame(i));
            token_ids[i] = it != state.pending.end() ? it->second : cfg_.blank_id;
            // The decoder commits the first non-blank only; later ones are re-asked.
            if (it != state.pending.end() && !committed) {
                state.pending.erase(it);
                committed = true;
            }
        }
    }

    void joint_tdt_argmax(RnntStreamState&, const float*, int, int, int32_t*, int32_t*) override {
        throw std::runtime_error("scripted engine does not implement TDT");
    }

    State& state(size_t i) { return *states_.at(i); }

   private:
    RnntConfig cfg_;
    std::vector<std::string> vocab_;
    std::vector<std::map<int64_t, int>> scripts_;
    std::mutex mu_;
    std::vector<State*> states_;
};

// The rule, independent of the decoder: a call is penalized iff its first frame
// is at least N frames past the last emitted token (or no token yet), and no
// call spans the frame where that changes.
bool
blank_penalty_calls_follow_rule(
    const std::vector<ScriptedRnntEngine::Call>& calls, const std::map<int64_t, int>& script,
    int after_frames, float penalty) {
    std::map<int64_t, int> pending = script;
    int64_t last = -1;
    for (const auto& c : calls) {
        const int64_t on = last < 0 ? 0 : last + after_frames;
        const bool penalized = c.first_frame >= on;
        if (!near(c.blank_bias, penalized ? -penalty : 0.0f))
            return false;
        if (!penalized && c.first_frame + c.frames > on)
            return false;
        const auto it = pending.lower_bound(c.first_frame);
        if (it != pending.end() && it->first < c.first_frame + c.frames) {
            last = it->first;
            pending.erase(it);
        }
    }
    return pending.empty();  // every scripted token was emitted
}

void
test_rnnt_blank_penalty_concurrent_streams() {
    constexpr int kStreams = 8;
    constexpr int kFrames = 64;
    constexpr int kAfter = 4;
    // Speech bursts and pauses of different lengths, offset per stream.
    std::vector<std::map<int64_t, int>> scripts(kStreams);
    for (int s = 0; s < kStreams; ++s)
        for (int64_t f : {1, 2, 3, 9, 10, 18, 19, 20, 21, 35, 36, 50, 55, 60})
            if (f + s < kFrames)
                scripts[s][f + s] = static_cast<int>((f + s) % 3);

    ScriptedRnntEngine eng(scripts);
    const int J = eng.rnnt_config().joint_dim;
    std::vector<std::unique_ptr<RnntGreedyDecoder>> decoders;
    for (int s = 0; s < kStreams; ++s)
        decoders.push_back(
            std::make_unique<RnntGreedyDecoder>(&eng, penalty_config(/*after_frames=*/kAfter)));

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> pool;
    for (int s = 0; s < kStreams; ++s) {
        pool.emplace_back([&, s]() {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            const int chunk = 1 + s % 5;  // chunk sizes differ between streams
            for (int64_t f0 = 0; f0 < kFrames; f0 += chunk) {
                const int T = static_cast<int>(std::min<int64_t>(chunk, kFrames - f0));
                std::vector<float> enc(static_cast<size_t>(T) * J, 0.0f);
                for (int i = 0; i < T; ++i)
                    enc[static_cast<size_t>(i) * J] = static_cast<float>(f0 + i);
                decoders[s]->step(enc.data(), J, T, f0);
            }
        });
    }
    while (ready.load() != kStreams) std::this_thread::yield();
    go.store(true);
    for (auto& thread : pool) thread.join();

    bool all = true;
    bool both_states = true;
    for (int s = 0; s < kStreams; ++s) {
        const auto& calls = eng.state(s).calls;
        all = all && blank_penalty_calls_follow_rule(calls, scripts[s], kAfter, kBlankPenalty);
        bool on = false, off = false;
        for (const auto& c : calls) (c.blank_bias != 0.0f ? on : off) = true;
        both_states = both_states && on && off;
    }
    check(both_states, "blank penalty: concurrent streams exercise both penalized and plain calls");
    check(all, "blank penalty: concurrent streams each follow the rule on every joint call");
}

}  // namespace

int
main() {
    test_sentencepiece_punctuation_detokenization();
    test_greedy_collapse_and_blank();
    test_greedy_cross_chunk_collapse();
    test_greedy_seam_dedup();
    test_greedy_last_emit_frame();
    test_greedy_confidence();
    test_greedy_word_timings();
    test_greedy_punctuation_word_timing();
    test_greedy_reset_utterance_vs_reset();
    test_greedy_open_word_needs_finalize();
    test_rnnt_emission_and_blank();
    test_rnnt_max_symbols_cap();
    test_rnnt_finalization_preserves_native_punctuation();
    test_rnnt_stalls_do_not_inject_punctuation();
    test_rnnt_late_punctuation_returns_to_previous_final();
    test_rnnt_endpoint_punctuation_continuity();
    test_rnnt_blank_penalty_after_silence();
    test_rnnt_blank_penalty_concurrent_streams();
    test_rnnt_state_threading_and_reset_utterance();
    test_rnnt_word_timings();
    test_rnnt_punctuation_word_timing();
    test_rnnt_open_word_needs_finalize();
    test_tdt_duration_and_state_commit();
    test_tdt_zero_duration_guard();
    test_tdt_duration_crosses_step_boundary();
    std::fprintf(stdout, g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
