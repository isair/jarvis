# 🧪 Jarvis Evaluation Report

**Generated:** 2026-01-26 01:45:59

## 📊 Model Comparison

This report compares eval results across officially supported models.
Use this to understand the performance tradeoffs when choosing a model.

| Metric | llama3.2:3b | gpt-oss:20b |
|--------|--------|--------|
| ✅ Passed | 68 | 79 |
| ❌ Failed | 1 | 0 |
| 🔸 Expected Fail | 6 | 0 |
| ⏭️ Skipped | 0 | 0 |
| 📊 Total | 82 | 82 |
| ⏱️ Duration | 145.0s | 757.3s |
| 📈 Pass Rate | 🟢 98.6% | 🟢 100.0% |

### Pass Rate Visualization

**llama3.2:3b:** 🟢 `███████████████████░` **98.6%**
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
| 3-turn conversation with topic changes | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Agent calls webSearch for info queries | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Agent chains search → fetch for details | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Agent recalls interests before personalized search (mocked) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Agent uses memory + nutrition data | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Bad: deflection without attempting answer | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Bad: empty acknowledgment | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Bad: generic greeting ignores query | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Enrichment results appear in system message | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Extraction with explicit quantities | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Follow-up references previous turn context | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Good: brief but informative | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Good: complete weekly forecast | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: bonjour (French) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: ciao (Italian) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: good morning | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: good night | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: goodbye | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: guten tag (German) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: hello | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: hey | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: hi there | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: hola (Spanish) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: how are you | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: merhaba (Turkish) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: ni hao (Chinese) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: thank you | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Greeting: thanks | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Handles ambiguous portion descriptions | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Instruction: be more brief | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Instruction: no emojis | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Instruction: prefer metric | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Instruction: remember Celsius | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Instruction: short version | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Instruction: speak in French | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Instruction: use Celsius | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| LLM uses enrichment, skips redundant recallConversation | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live greeting: bonjour (French) | ⚠️ 1/1 (100%) | ✅ 3/3 (100%) |
| Live greeting: hello | 🔸 3/3 XFAIL | ✅ 3/3 (100%) |
| Live greeting: how are you | 🔸 3/3 XFAIL | ✅ 3/3 (100%) |
| Live greeting: ni hao (Chinese) | 🔸 3/3 XFAIL | ✅ 3/3 (100%) |
| Live instruction: be more brief | 🔸 3/3 XFAIL | ✅ 3/3 (100%) |
| Live instruction: remember Celsius | 🔸 3/3 XFAIL | ✅ 3/3 (100%) |
| Live instruction: use Celsius | 🔸 3/3 XFAIL | ✅ 3/3 (100%) |
| Live weather query with real LLM | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live: LLM checks memory before asking about interests | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Live: weather query triggers tools | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Location context flows to search queries | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| LogMealTool stores meals with macros | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Memory enrichment: personalized news | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Memory enrichment: personalized restaurant | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Memory enrichment: time-based recall | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Memory enrichment: topic recall | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: None \u2192 LARGE (default) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: empty \u2192 LARGE (default) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: gemma2:27b \u2192 LARGE | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: gpt-oss:20b \u2192 LARGE | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: llama3.1:8b \u2192 LARGE | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: llama3.2:1b \u2192 SMALL | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: llama3.2:3b \u2192 SMALL | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: mistral:7b \u2192 SMALL | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Model size: qwen2.5:14b \u2192 LARGE | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: caesar salad with chicken | ✅ 3/3 (100%) | ⚠️ 2/3 (67%) |
| Nutrition: cheeseburger with fries | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: chicken with broccoli | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: oatmeal with banana | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: pepperoni pizza slice | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Nutrition: protein shake | ✅ 3/3 (100%) | ⚠️ 2/3 (67%) |
| Nutrition: scrambled eggs with toast | ✅ 3/3 (100%) | ⚠️ 2/3 (67%) |
| Nutrition: spaghetti bolognese | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Rapid back-and-forth topic switching | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Returns NONE for non-food inputs | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Returns valid JSON with all required fields | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Simple meal baseline (2 boiled eggs) | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool query: meal recall | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool query: news search | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool query: weather | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool query: weather with location | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Tool query: web search | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Topic switch: search → weather uses getWeather | ✅ 3/3 (100%) | ✅ 3/3 (100%) |
| Topic switch: weather → restaurant uses webSearch | ⚠️ 2/3 (67%) | ✅ 3/3 (100%) |
| Topic switch: weather → store hours uses webSearch | ❌ 0/3 (0%) | ✅ 3/3 (100%) |

---

## 🎤 Intent Judge Tests

> These tests evaluate the voice intent classification system.
> They use a fixed model (`llama3.2:3b`) and are not part of the model comparison.

**Model:** `llama3.2:3b` (fixed)
**Results:** 4 passed, 0 failed, 0 expected failures

| Test Case | Pass Rate | Status |
|-----------|-----------|--------|
| test_hot_window_mode_indicated_in_prompt | 3/3 (100%) | ✅ |
| test_returns_none_when_ollama_unavailable | 3/3 (100%) | ✅ |
| test_system_prompt_has_echo_guidance | 3/3 (100%) | ✅ |
| test_tts_text_included_for_echo_detection | 3/3 (100%) | ✅ |

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
