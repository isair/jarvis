# 🧪 Jarvis Evaluation Report

**Generated:** 2026-01-25 22:11:48

## 📊 Model Comparison

This report compares eval results across officially supported models.
Use this to understand the performance tradeoffs when choosing a model.

| Metric | llama3.2:3b | gpt-oss:20b |
|--------|--------|--------|
| ✅ Passed | 266 | 288 |
| ❌ Failed | 11 | 0 |
| 🔸 Expected Fail | 26 | 15 |
| ⏭️ Skipped | 0 | 0 |
| 📊 Total | 303 | 303 |
| ⏱️ Duration | 460.9s | 1997.2s |
| 📈 Pass Rate | 🟢 96.0% | 🟢 100.0% |

### Pass Rate Visualization

**llama3.2:3b:** 🟢 `███████████████████░` **96.0%**
**gpt-oss:20b:** 🟢 `████████████████████` **100.0%**

### 💡 Model Selection Guide

| Model | Best For | Trade-offs |
|-------|----------|------------|
| `llama3.2:3b` | Quick responses, lower RAM usage | May struggle with complex reasoning |
| `gpt-oss:20b` | Best accuracy, complex tasks | Slower, requires more RAM |

---

## 📋 Detailed Test Results

| Test Case | llama3.2:3b | gpt-oss:20b |
|-----------|----------|----------|
| **Response Quality** | | |
| Response quality: good complete forecast | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Response quality: good brief response | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Response quality: bad generic greeting | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Response quality: bad deflection | ❌ 0/3 (0%) | ✅ 3/3 (100%) |
| Response quality: bad empty acknowledgment | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| **Context & Tool Usage** | | |
| Location context in search | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Simple search flow | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool chaining: search then fetch | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| **Multi-Step Reasoning** | | |
| Nutrition advice uses memory and data | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Personalized news uses memory | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| **Memory Enrichment** | | |
| Memory enrichment: personalized news | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Memory enrichment: personalized restaurant | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Memory enrichment: topic recall | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Memory enrichment: time-based recall | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Enrichment provides context to LLM | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| LLM uses enrichment without redundant tool call | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| **Live End-to-End** | | |
| Live: weather query | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live: personalized query recalls memory | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| **Model Size Detection** | | |
| Model size: llama3.2:3b → SMALL | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: llama3.2:1b → SMALL | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: mistral:7b → SMALL | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: gpt-oss:20b → LARGE | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: llama3.1:8b → LARGE | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: qwen2.5:14b → LARGE | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: gemma2:27b → LARGE | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: None → LARGE (default) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: empty → LARGE (default) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| **Greeting Recognition (Mocked)** | | |
| Greeting: hello | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: hi there | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: hey | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: ni hao (Chinese) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: bonjour (French) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: hola (Spanish) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: merhaba (Turkish) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: ciao (Italian) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: guten tag (German) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: how are you | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: thank you | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: thanks | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: goodbye | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: good morning | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: good night | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| **Tool-Triggering Queries (Mocked)** | | |
| Tool query: weather | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool query: web search | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool query: weather with location | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool query: news search | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool query: meal recall | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| **Greeting Recognition (Live)** | | |
| Live greeting: hello | 🔸 3/3 XFAIL | ✅ 3/3 (100%) |
| Live greeting: ni hao (Chinese) | ⚠️ 1/3 (33%) | ✅ 3/3 (100%) |
| Live greeting: bonjour (French) | 🔸 3/3 XFAIL | ✅ 3/3 (100%) |
| Live greeting: how are you | 🔸 3/3 XFAIL | ✅ 3/3 (100%) |
| Live: weather triggers tools | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| **Multi-Turn Context** | | |
| Topic switch: weather → store hours | ❌ 0/3 (0%) | ✅ 3/3 (100%) |
| Topic switch: weather → restaurant | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Topic switch: search → weather | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Follow-up references previous context | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Three-turn topic changes | ❌ 0/3 (0%) | ✅ 3/3 (100%) |
| Rapid topic switching | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| **Nutrition Extraction** | | |
| Nutrition: chicken with broccoli | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: scrambled eggs with toast | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: pepperoni pizza slice | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: oatmeal with banana | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: cheeseburger with fries | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: caesar salad with chicken | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Nutrition: protein shake | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: spaghetti bolognese | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: valid JSON structure | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: handles ambiguous portions | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: rejects non-food | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition tool: extracts macros | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: simple meal extraction | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: extraction with quantities | ✅ 3/3 (100%) | ✅ 3/3 (100%) |

---

## 🎤 Intent Judge Tests

> These tests evaluate the voice intent classification system.
> They use a fixed model (`llama3.2:3b`) and are not part of the model comparison.

**Model:** `llama3.2:3b` (fixed)
**Results:** 51 passed, 0 failed, 15 expected failures

| Test Case | Pass Rate | Status |
|-----------|-----------|--------|
| wake_word_simple_question | 3/3 (100%) | ✅ |
| wake_word_with_pre_chatter | 3/3 (100%) | ✅ |
| pure_echo_rejected | 3/3 XFAIL | 🔸 |
| partial_echo_rejected | 3/3 XFAIL | 🔸 |
| echo_plus_followup_extracted | 3/3 (100%) | ✅ |
| echo_plus_different_query | 3/3 (100%) | ✅ |
| stop_command_during_tts | 3/3 (100%) | ✅ |
| quiet_command | 3/3 (100%) | ✅ |
| mentioned_in_narrative_past_tense | 3/3 XFAIL | 🔸 |
| mentioned_in_narrative_third_person | 3/3 XFAIL | 🔸 |
| hot_window_simple_followup | 3/3 (100%) | ✅ |
| hot_window_thanks_followup | 3/3 (100%) | ✅ |
| non_english_followup | 3/3 XFAIL | 🔸 |
| wake_word_different_topic_not_echo | 3/3 (100%) | ✅ |
| wake_word_completely_unrelated_to_tts | 3/3 (100%) | ✅ |
| hot_window_mode_indicated_in_prompt | 3/3 (100%) | ✅ |
| tts_text_included_for_echo_detection | 3/3 (100%) | ✅ |
| system_prompt_has_echo_guidance | 3/3 (100%) | ✅ |
| returns_none_when_ollama_unavailable | 3/3 (100%) | ✅ |
| buffer_with_echoes_then_wake_word_query | 3/3 (100%) | ✅ |
| echo_plus_rejected_similar_plus_wake_retry | 3/3 (100%) | ✅ |
| full_buffer_with_tts_echoes_and_wake_retry | 3/3 (100%) | ✅ |
| echo_slipped_through_then_wake_query | 3/3 (100%) | ✅ |
| buffer_echo_then_followup_hot_window | 3/3 (100%) | ✅ |
| multiple_echoes_then_interrupt | 3/3 (100%) | ✅ |
| multi_person_weather_discussion | 3/3 (100%) | ✅ |
| multi_person_restaurant_recommendation | 3/3 (100%) | ✅ |
| multi_person_travel_planning | 3/3 (100%) | ✅ |
| multi_person_vague_reference | 3/3 (100%) | ✅ |

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
