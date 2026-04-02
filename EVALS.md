# 🧪 Jarvis Evaluation Report

**Generated:** 2026-04-03 00:55:49

## 📊 Model Comparison

This report compares eval results across officially supported models.
Use this to understand the performance tradeoffs when choosing a model.

| Metric | gemma4:e2b | gpt-oss:20b |
|--------|--------|--------|
| ✅ Passed | 38 | 46 |
| ❌ Failed | 2 | 0 |
| 🔸 Expected Fail | 0 | 0 |
| ⏭️ Skipped | 0 | 0 |
| 📊 Total | 46 | 46 |
| ⏱️ Duration | 405.6s | 249.5s |
| 📈 Pass Rate | 🟢 95.0% | 🟢 100.0% |

### Pass Rate Visualization

**gemma4:e2b:** 🟢 `███████████████████░` **95.0%**
**gpt-oss:20b:** 🟢 `████████████████████` **100.0%**

### 💡 Model Selection Guide

| Model | Best For | Trade-offs |
|-------|----------|------------|
| `gemma4` | Quick responses, lower RAM usage | May struggle with complex reasoning |
| `gpt-oss:20b` | Best accuracy, complex tasks | Slower, requires more RAM |

---

## 📋 Detailed Test Results

| Test Case | gemma4:e2b | gpt-oss:20b |
|-----------|----------|----------|
| 3-turn conversation with topic changes | ❌ 0/3 (0%) | ✅ 3/3 (100%) |
| Agent calls webSearch for info queries | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Agent chains search → fetch for details | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Agent recalls interests before personalized search (mocked) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Agent uses memory + nutrition data | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Bad: deflection without attempting answer | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Bad: empty acknowledgment | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Bad: generic greeting ignores query | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Enrichment results appear in system message | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Extraction with explicit quantities | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Follow-up references previous turn context | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Good: brief but informative | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Good: complete weekly forecast | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Handles ambiguous portion descriptions | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| LLM uses enrichment, skips redundant recallConversation | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live greeting: bonjour (French) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live greeting: hello | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live greeting: how are you | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live greeting: ni hao (Chinese) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live instruction: be more brief | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live instruction: remember Celsius | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live instruction: use Celsius | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live weather query with real LLM | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Live: LLM checks memory before asking about interests | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live: weather query triggers tools | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Location context flows to search queries | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| LogMealTool stores meals with macros | ❌ 0/3 (0%) | ✅ 3/3 (100%) |
| Memory enrichment: personalized news | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Memory enrichment: personalized restaurant | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Memory enrichment: time-based recall | ⚠️ 1/3 (33%) | ✅ 3/3 (100%) |
| Memory enrichment: topic recall | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Nutrition: caesar salad with chicken | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: cheeseburger with fries | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: chicken with broccoli | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: oatmeal with banana | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: pepperoni pizza slice | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: protein shake | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: scrambled eggs with toast | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: spaghetti bolognese | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Rapid back-and-forth topic switching | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Returns NONE for non-food inputs | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Returns valid JSON with all required fields | ⚠️ 1/3 (33%) | ✅ 3/3 (100%) |
| Simple meal baseline (2 boiled eggs) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Topic switch: search → weather uses getWeather | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Topic switch: weather → restaurant uses webSearch | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Topic switch: weather → store hours uses webSearch | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |

---

## 🎤 Intent Judge Tests

> These tests evaluate the voice intent classification system.
> They use a fixed model (`gemma4`) and are not part of the model comparison.

**Model:** `gemma4` (fixed)
**Results:** 4 passed, 22 failed, 13 expected failures

| Test Case | Pass Rate | Status |
|-----------|-----------|--------|
| ambient_speech_then_wake_word | 0/3 (0%) | ❌ |
| buffer_echo_then_followup_hot_window | 0/3 (0%) | ❌ |
| buffer_with_echoes_then_wake_word_query | 0/3 (0%) | ❌ |
| context_synthesis_food_opinion | 0/3 (0%) | ❌ |
| context_synthesis_movie_question | 0/3 (0%) | ❌ |
| context_synthesis_single_utterance | 0/3 (0%) | ❌ |
| context_synthesis_weather_opinion | 0/3 (0%) | ❌ |
| context_synthesis_with_prior_ambient | 0/3 (0%) | ❌ |
| echo_plus_different_query | 0/3 (0%) | ❌ |
| echo_plus_followup_extracted | 0/3 (0%) | ❌ |
| echo_plus_rejected_similar_plus_wake_retry | 0/3 (0%) | ❌ |
| echo_slipped_through_then_wake_query | 0/3 (0%) | ❌ |
| full_buffer_with_tts_echoes_and_wake_retry | 0/3 (0%) | ❌ |
| hot_window_simple_followup | 1/3 (33%) | ⚠️ |
| hot_window_thanks_followup | 0/3 (0%) | ❌ |
| mentioned_in_narrative_past_tense | 3/3 XFAIL | 🔸 |
| mentioned_in_narrative_third_person | 3/3 XFAIL | 🔸 |
| multi_person_restaurant_recommendation | 3/3 XFAIL | 🔸 |
| multi_person_travel_planning | 0/3 (0%) | ❌ |
| multi_person_vague_reference | 3/3 XFAIL | 🔸 |
| multi_person_weather_discussion | 3/3 XFAIL | 🔸 |
| multiple_echoes_then_interrupt | 3/3 XFAIL | 🔸 |
| no_wake_word_casual_speech | 0/3 (0%) | ❌ |
| no_wake_word_in_buffer | 3/3 XFAIL | 🔸 |
| no_wake_word_simple_question | 3/3 XFAIL | 🔸 |
| non_english_followup | 3/3 XFAIL | 🔸 |
| partial_echo_rejected | 3/3 XFAIL | 🔸 |
| pure_echo_rejected | 3/3 XFAIL | 🔸 |
| quiet_command | 3/3 XFAIL | 🔸 |
| stop_command_during_tts | 0/3 (0%) | ❌ |
| test_hot_window_mode_indicated_in_prompt | 3/3 (100%) | ✅ |
| test_processed_segment_not_reextracted | 1/3 (33%) | ⚠️ |
| test_returns_none_when_ollama_unavailable | 3/3 (100%) | ✅ |
| test_system_prompt_has_echo_guidance | 3/3 (100%) | ✅ |
| test_tts_text_included_for_echo_detection | 3/3 (100%) | ✅ |
| user_followup_acknowledgment_statement | 3/3 XFAIL | 🔸 |
| user_followup_statement_after_opinion_question | 0/3 (0%) | ❌ |
| user_followup_statement_after_question_nihilism | 0/3 (0%) | ❌ |
| user_followup_statement_meaning_of_life | 0/3 (0%) | ❌ |
| wake_word_completely_unrelated_to_tts | 1/3 (33%) | ⚠️ |
| wake_word_different_topic_not_echo | 0/3 (0%) | ❌ |
| wake_word_simple_question | 2/3 (67%) | ⚠️ |
| wake_word_with_pre_chatter | 0/3 (0%) | ❌ |

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

---

*Report generated by Jarvis eval suite*