# 🧪 Jarvis Evaluation Report

**Generated:** 2026-04-18 13:19:13

## 📊 TL;DR

**Overall:** 🟢 **158/166 passed (95.2%)** across all categories

| Category | Model | Passed | Failed | Skipped | Pass Rate |
|----------|-------|-------:|-------:|--------:|----------:|
| 🤖 Agent behaviour | `gemma4:e2b` | 56 | 7 | 0 | 🟢 88.9% |
| 🤖 Agent behaviour | `gpt-oss:20b` | 65 | 1 | 0 | 🟢 98.5% |
| 🎤 Intent judge | `gemma4:e2b` (fixed) | 32 | 0 | 0 | 🟢 100.0% |
| 🔍 Tool selection | `nomic-embed-text` (fixed) | 5 | 0 | 0 | 🟢 100.0% |

### 💡 Model Selection Guide

| Model | Best For | Trade-offs |
|-------|----------|------------|
| `gemma4:e2b` | Quick responses, lower RAM usage | May struggle with complex reasoning |
| `gpt-oss:20b` | Best accuracy, complex tasks | Slower, requires more RAM |

---

## 🤖 Agent behaviour

> Runs the full agent pipeline against each judge model. Tests are compared side-by-side.

| Test Case | gemma4:e2b | gpt-oss:20b |
|-----------|----------:|----------:|
| 3-turn conversation with topic changes | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Agent calls webSearch for info queries | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Agent chains search → fetch for details | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Agent recalls interests before personalized search (mocked) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Agent uses memory + nutrition data | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Bad: deflection without attempting answer | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Bad: empty acknowledgment | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Bad: generic greeting ignores query | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Diet changed from bulking to cutting | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Enrichment results appear in system message | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Extraction with explicit quantities | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Follow-up references previous turn context | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Good: brief but informative | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Good: complete weekly forecast | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Handles ambiguous portion descriptions | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| LLM uses enrichment, skips redundant recallConversation | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live greeting: hello | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live greeting: ni hao (Chinese) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live instruction: be more brief | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live instruction: use Celsius | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live weather query with real LLM | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Live: LLM checks memory before asking about interests | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Live: weather query triggers tools | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Location context flows to search queries | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| LogMealTool stores meals with macros | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Memory enrichment: personalized news | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Memory enrichment: time-based recall | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Memory enrichment: topic recall | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No deflection: tech news | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No deflection: time query | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| No deflection: tomorrow weather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| No deflection: weekly rain forecast | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: local business details and user location | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: non-English summary (Turkish) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: relocation plans and employment | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Novel knowledge: user diet plan and preferred recipe | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nutrition: cheeseburger with fries | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nutrition: chicken with broccoli | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Nutrition: oatmeal with banana | ✅ 1/1 (100%) | ❌ 0/1 (0%) |
| Office days changed from Mon/Wed to Mon/Thu | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reframing: life events framed as facts with temporal context | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reframing: requests become knowledge, not interaction descriptions | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reject: assistant self-references (recommendations are not knowledge) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Reject: stale temporal snapshots (weather, time of day) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Returns NONE for non-food inputs | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Returns valid JSON with all required fields | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| Simple meal baseline (2 boiled eggs) | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Tool retry: explicit tool mention | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Tool retry: vague go ahead | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Tool retry: vague just try | 🔸 1/1 XFAIL | ✅ 1/1 (100%) |
| Topic switch: search → weather uses getWeather | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| Topic switch: weather → store hours uses webSearch | ❌ 0/1 (0%) | ✅ 1/1 (100%) |
| test_active_hot_window_follow_up_accepted | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_casual_statement_without_wake_word_rejected | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_judge_echo_claim_overridden_in_hot_window | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_judge_empty_conversation_returns_empty | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_judge_mixed_summary_filters_noise | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_no_wake_word_rejected_despite_judge | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_raw_text_preserved_in_hot_window | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_speech_long_after_tts_requires_wake_word | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_stop_during_tts_interrupts_immediately | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_tts_echo_segments_skipped_user_query_extracted | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_user_query_not_confused_with_echo_after_tts | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_utterance_started_during_tts_treated_as_hot_window | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_wake_word_query_after_echo_segments | ✅ 1/1 (100%) | ✅ 1/1 (100%) |
| test_wake_word_query_uses_judge_extraction | ✅ 1/1 (100%) | ✅ 1/1 (100%) |

---

## 🎤 Intent judge

> Pinned to `gemma4:e2b` (the voice intent classifier). Not affected by the judge model.

| Test Case | Pass Rate | Status |
|-----------|-----------|:------:|
| alias_after_narrative_context | 1/1 (100%) | ✅ |
| alias_treated_as_wake_word | 1/1 (100%) | ✅ |
| buffer_echo_then_followup_hot_window | 1/1 (100%) | ✅ |
| context_synthesis_weather_opinion | 1/1 (100%) | ✅ |
| context_synthesis_with_prior_ambient | 1/1 (100%) | ✅ |
| cross_segment_answer_that_weather | 1/1 (100%) | ✅ |
| cross_segment_answer_that_with_noise | 1/1 (100%) | ✅ |
| cross_segment_answered_that_whisper_variant | 1/1 (100%) | ✅ |
| cross_segment_dinosaur_opinion | 1/1 (100%) | ✅ |
| cross_segment_go_ahead_and_answer | 1/1 (100%) | ✅ |
| cross_segment_hot_window_followup | 1/1 (100%) | ✅ |
| cross_segment_imperative_superseded_by_new_question | 1/1 (100%) | ✅ |
| echo_plus_followup_extracted | 1/1 (100%) | ✅ |
| echo_plus_rejected_similar_plus_wake_retry | 1/1 (100%) | ✅ |
| hot_window_simple_followup | 1/1 (100%) | ✅ |
| mentioned_in_narrative_past_tense | 1/1 (100%) | ✅ |
| multi_person_vague_reference | 1/1 (100%) | ✅ |
| multi_person_weather_discussion | 1/1 (100%) | ✅ |
| multiple_echoes_then_interrupt | 1/1 (100%) | ✅ |
| no_wake_word_casual_speech | 1/1 (100%) | ✅ |
| no_wake_word_in_buffer | 1/1 (100%) | ✅ |
| stop_command_during_tts | 1/1 (100%) | ✅ |
| test_hot_window_mode_indicated_in_prompt | 1/1 (100%) | ✅ |
| test_old_query_not_re_extracted | 1/1 (100%) | ✅ |
| test_processed_segment_not_reextracted | 1/1 (100%) | ✅ |
| test_returns_none_when_ollama_unavailable | 1/1 (100%) | ✅ |
| test_system_prompt_has_echo_guidance | 1/1 (100%) | ✅ |
| test_tts_text_included_for_echo_detection | 1/1 (100%) | ✅ |
| user_followup_statement_after_question_nihilism | 1/1 (100%) | ✅ |
| wake_word_command_timer | 1/1 (100%) | ✅ |
| wake_word_simple_question | 1/1 (100%) | ✅ |
| wake_word_statement_remember | 1/1 (100%) | ✅ |

---

## 🔍 Tool selection

> Pinned to `nomic-embed-text` (embedding-based filter). Not affected by the judge model.

| Test Case | Pass Rate | Status |
|-----------|-----------|:------:|
| location weather query selects getWeather and few others | 1/1 (100%) | ✅ |
| meal logging selects logMeal and few others | 1/1 (100%) | ✅ |
| meal recall selects fetchMeals and few others | 1/1 (100%) | ✅ |
| weather query selects getWeather and few others | 1/1 (100%) | ✅ |
| web search query selects webSearch and few others | 1/1 (100%) | ✅ |

---

### 📖 Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully passed (100% pass rate) |
| ⚠️ | Partial pass (some runs failed) |
| ❌ | Fully failed (0% pass rate) |
| ⏭️ | Skipped (missing dependencies) |
| 🔸 | Expected failure (known limitation) |
| 🎉 | Unexpectedly passed (bug fixed!) |
| ➖ | Not run for this model |

*Report generated by Jarvis eval suite*