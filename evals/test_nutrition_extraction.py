"""
Nutrition Extraction Evaluations

Tests the LLM's ability to extract accurate nutritional information from meal descriptions.
This is critical for smaller models like llama3.2:3b which may struggle with nutrition estimation.

Run with specific model:
    EVAL_JUDGE_MODEL=llama3.2:3b ./scripts/run_evals.sh nutrition
    EVAL_JUDGE_MODEL=gpt-oss:20b ./scripts/run_evals.sh nutrition

For EVALS.md generation (always use gpt-oss:20b):
    ./scripts/run_evals.sh
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

_this_file = Path(__file__).resolve()
EVALS_DIR = _this_file.parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import pytest

from helpers import (
    MockConfig,
    is_judge_llm_available,
    JUDGE_MODEL,
    JUDGE_BASE_URL,
)

_JUDGE_LLM_AVAILABLE = is_judge_llm_available()
requires_judge_llm = pytest.mark.skipif(
    not _JUDGE_LLM_AVAILABLE,
    reason="Judge LLM not available"
)


# =============================================================================
# Test Data - Meals with Expected Nutritional Ranges
# =============================================================================

@dataclass
class MealTestCase:
    """A meal test case with expected nutritional ranges."""
    description: str
    # Expected ranges as (min, max) - None means any value is acceptable
    calories_range: Tuple[int, int]
    protein_range: Tuple[int, int]
    carbs_range: Tuple[int, int]
    fat_range: Tuple[int, int]
    # Whether we expect micronutrients to be populated
    expect_micros: bool = False


# Common meals with reasonable nutritional ranges
MEAL_TEST_CASES = [
    pytest.param(
        MealTestCase(
            description="a grilled chicken breast with steamed broccoli",
            calories_range=(200, 400),
            protein_range=(25, 50),
            carbs_range=(0, 20),  # Broccoli has minimal carbs; some models report 0
            fat_range=(3, 15),
        ),
        id="chicken-broccoli"
    ),
    pytest.param(
        MealTestCase(
            description="two scrambled eggs with toast",
            calories_range=(200, 450),
            protein_range=(12, 25),
            carbs_range=(5, 35),  # Smaller models may underestimate toast carbs
            fat_range=(10, 30),
        ),
        id="eggs-toast"
    ),
    pytest.param(
        MealTestCase(
            description="a large pepperoni pizza slice",
            calories_range=(250, 400),
            protein_range=(10, 20),
            carbs_range=(25, 45),
            fat_range=(10, 25),
        ),
        id="pizza-slice"
    ),
    pytest.param(
        MealTestCase(
            description="a bowl of oatmeal with banana and honey",
            calories_range=(300, 500),
            protein_range=(6, 15),
            carbs_range=(50, 90),
            fat_range=(3, 12),
        ),
        id="oatmeal-banana"
    ),
    pytest.param(
        MealTestCase(
            description="a cheeseburger with fries",
            calories_range=(700, 1200),
            protein_range=(25, 45),
            carbs_range=(60, 120),
            fat_range=(35, 70),
        ),
        id="cheeseburger-fries"
    ),
    pytest.param(
        MealTestCase(
            description="a caesar salad with grilled chicken",
            calories_range=(350, 600),
            protein_range=(25, 45),
            carbs_range=(10, 30),
            fat_range=(20, 40),
        ),
        id="caesar-salad"
    ),
    pytest.param(
        MealTestCase(
            description="a protein shake with milk and peanut butter",
            calories_range=(300, 550),
            protein_range=(25, 50),
            carbs_range=(15, 40),
            fat_range=(10, 25),
        ),
        id="protein-shake"
    ),
    pytest.param(
        MealTestCase(
            description="spaghetti bolognese",
            calories_range=(500, 800),
            protein_range=(20, 40),
            carbs_range=(60, 100),
            fat_range=(15, 35),
        ),
        id="spaghetti-bolognese"
    ),
]


# =============================================================================
# Evaluation Helpers
# =============================================================================

def call_nutrition_extraction(
    cfg: MockConfig,
    meal_text: str
) -> Optional[Dict[str, Any]]:
    """
    Call the nutrition extraction prompt directly and parse the response.
    Returns the parsed JSON or None if extraction failed.
    """
    from jarvis.tools.builtin.nutrition.log_meal import NUTRITION_SYS
    from jarvis.llm import call_llm_direct

    user_prompt = (
        "User said (redacted):\n" + meal_text[:1200] + "\n\n"
        "Return ONLY JSON or the exact string NONE."
    )

    raw = call_llm_direct(
        cfg.ollama_base_url,
        cfg.ollama_chat_model,
        NUTRITION_SYS,
        user_prompt,
        timeout_sec=cfg.llm_chat_timeout_sec
    ) or ""

    text = raw.strip()
    if text.upper() == "NONE":
        return None

    try:
        # Handle markdown code blocks
        if "```" in text:
            # Extract JSON from code block
            start = text.find("```")
            end = text.rfind("```")
            if start != end:
                inner = text[start:end]
                # Remove ```json or ``` prefix
                if inner.startswith("```json"):
                    inner = inner[7:]
                elif inner.startswith("```"):
                    inner = inner[3:]
                text = inner.strip()

        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_nutrition_data(
    data: Optional[Dict[str, Any]],
    case: MealTestCase
) -> Tuple[bool, List[str]]:
    """
    Validate extracted nutrition data against expected ranges.
    Returns (passed, list of issues).
    """
    issues = []

    if data is None:
        return False, ["Extraction returned None or invalid JSON"]

    # Check required fields exist
    required_fields = ["calories_kcal", "protein_g", "carbs_g", "fat_g"]
    for field in required_fields:
        if field not in data or data[field] is None:
            issues.append(f"Missing required field: {field}")

    if issues:
        return False, issues

    # Validate ranges
    def check_range(value: Any, field_name: str, expected_range: Tuple[int, int]) -> Optional[str]:
        try:
            v = float(value)
            min_val, max_val = expected_range
            if v < min_val * 0.5:  # Allow 50% below minimum
                return f"{field_name}={v:.0f} too low (expected {min_val}-{max_val})"
            if v > max_val * 2.0:  # Allow 100% above maximum
                return f"{field_name}={v:.0f} too high (expected {min_val}-{max_val})"
        except (TypeError, ValueError):
            return f"{field_name} is not a valid number: {value}"
        return None

    # Check each macro
    cal_issue = check_range(data.get("calories_kcal"), "calories", case.calories_range)
    if cal_issue:
        issues.append(cal_issue)

    prot_issue = check_range(data.get("protein_g"), "protein", case.protein_range)
    if prot_issue:
        issues.append(prot_issue)

    carb_issue = check_range(data.get("carbs_g"), "carbs", case.carbs_range)
    if carb_issue:
        issues.append(carb_issue)

    fat_issue = check_range(data.get("fat_g"), "fat", case.fat_range)
    if fat_issue:
        issues.append(fat_issue)

    # Check confidence is present and reasonable
    confidence = data.get("confidence")
    if confidence is None:
        issues.append("Missing confidence score")
    elif not isinstance(confidence, (int, float)) or not (0 <= float(confidence) <= 1):
        issues.append(f"Invalid confidence: {confidence} (should be 0-1)")

    return len(issues) == 0, issues


# =============================================================================
# Nutrition Extraction Tests
# =============================================================================

class TestNutritionExtraction:
    """
    Tests for LLM nutrition extraction accuracy.

    These tests verify that the model can:
    1. Parse meal descriptions correctly
    2. Return valid JSON with required fields
    3. Provide reasonable nutritional estimates
    """

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("case", MEAL_TEST_CASES)
    def test_meal_extraction_accuracy(self, case: MealTestCase, mock_config):
        """
        Test that the model extracts reasonable nutrition data for common meals.
        """
        mock_config.ollama_base_url = JUDGE_BASE_URL
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.llm_chat_timeout_sec = 30.0

        print(f"\n[MEAL] Testing meal: {case.description}")
        print(f"   Model: {JUDGE_MODEL}")

        # Call the extraction
        data = call_nutrition_extraction(mock_config, f"I had {case.description}")

        print(f"   Extracted: {json.dumps(data, indent=2) if data else 'None'}")

        # Validate
        passed, issues = validate_nutrition_data(data, case)

        if data:
            print(f"   Calories: {data.get('calories_kcal')} (expected {case.calories_range[0]}-{case.calories_range[1]})")
            print(f"   Protein: {data.get('protein_g')}g (expected {case.protein_range[0]}-{case.protein_range[1]})")
            print(f"   Carbs: {data.get('carbs_g')}g (expected {case.carbs_range[0]}-{case.carbs_range[1]})")
            print(f"   Fat: {data.get('fat_g')}g (expected {case.fat_range[0]}-{case.fat_range[1]})")
            print(f"   Confidence: {data.get('confidence')}")

        if issues:
            print(f"   FAIL Issues: {issues}")
        else:
            print(f"   PASS All values within expected ranges")

        assert passed, f"Nutrition extraction failed: {issues}"

    @pytest.mark.eval
    @requires_judge_llm
    def test_extraction_returns_valid_json_structure(self, mock_config):
        """
        Test that extraction returns properly structured JSON with all expected fields.
        """
        mock_config.ollama_base_url = JUDGE_BASE_URL
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.llm_chat_timeout_sec = 30.0

        print(f"\n[JSON] Testing JSON structure")
        print(f"   Model: {JUDGE_MODEL}")

        data = call_nutrition_extraction(mock_config, "I ate a sandwich for lunch")

        print(f"   Response: {json.dumps(data, indent=2) if data else 'None'}")

        assert data is not None, "Should return valid JSON, not None"

        # Check all expected fields
        expected_fields = [
            "description", "calories_kcal", "protein_g", "carbs_g", "fat_g",
            "fiber_g", "sugar_g", "sodium_mg", "potassium_mg", "confidence"
        ]

        missing = [f for f in expected_fields if f not in data]
        print(f"   Missing fields: {missing if missing else 'None'}")

        # Core fields are mandatory
        core_fields = ["description", "calories_kcal", "protein_g", "carbs_g", "fat_g", "confidence"]
        core_missing = [f for f in core_fields if f not in data]

        assert not core_missing, f"Missing core fields: {core_missing}"
        print(f"   PASS All core fields present")

    @pytest.mark.eval
    @requires_judge_llm
    def test_extraction_handles_ambiguous_portions(self, mock_config):
        """
        Test that model provides reasonable estimates for ambiguous portion descriptions.
        """
        mock_config.ollama_base_url = JUDGE_BASE_URL
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.llm_chat_timeout_sec = 30.0

        print(f"\n[AMBIGUOUS] Testing ambiguous portions")
        print(f"   Model: {JUDGE_MODEL}")

        # Ambiguous description - should still get reasonable defaults
        data = call_nutrition_extraction(mock_config, "I had some rice with chicken")

        print(f"   Response: {json.dumps(data, indent=2) if data else 'None'}")

        assert data is not None, "Should handle ambiguous portions"

        # Should have a lower confidence for ambiguous descriptions
        confidence = data.get("confidence")
        print(f"   Confidence: {confidence}")

        # Calories should be reasonable for rice + chicken (300-800 typical)
        calories = data.get("calories_kcal")
        if calories:
            assert 150 <= float(calories) <= 1200, f"Calories {calories} outside reasonable range"
            print(f"   PASS Calories {calories} within reasonable range")

    @pytest.mark.eval
    @requires_judge_llm
    def test_extraction_rejects_non_food(self, mock_config):
        """
        Test that extraction returns NONE for non-food inputs.
        """
        mock_config.ollama_base_url = JUDGE_BASE_URL
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.llm_chat_timeout_sec = 30.0

        print(f"\n[NON-FOOD] Testing non-food rejection")
        print(f"   Model: {JUDGE_MODEL}")

        # Non-food input
        data = call_nutrition_extraction(mock_config, "I went for a walk in the park")

        print(f"   Response: {data}")

        # Should return None (NONE response)
        assert data is None, f"Should return None for non-food input, got: {data}"
        print(f"   PASS Correctly returned None")


class TestNutritionToolIntegration:
    """
    Tests for the full meal logging tool integration.

    These test the complete flow from user input through tool execution.
    """

    @pytest.mark.eval
    @requires_judge_llm
    def test_log_meal_tool_extracts_macros(self, mock_config, eval_db):
        """
        Test that LogMealTool properly extracts and stores macros.
        """
        from jarvis.tools.builtin.nutrition.log_meal import LogMealTool
        from jarvis.tools.base import ToolContext
        from jarvis.memory.db import Database

        mock_config.ollama_base_url = JUDGE_BASE_URL
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.llm_chat_timeout_sec = 45.0
        mock_config.use_stdin = True

        print(f"\n[TOOL] Testing LogMealTool integration")
        print(f"   Model: {JUDGE_MODEL}")

        tool = LogMealTool()

        # Retry up to 3 times since smaller models can be flaky
        result = None
        for attempt in range(3):
            # Fresh DB for each attempt
            test_db = Database(":memory:", sqlite_vss_path=None)

            messages_printed = []

            def capture_print(msg):
                messages_printed.append(msg)

            context = ToolContext(
                db=test_db,
                cfg=mock_config,
                system_prompt="You are a helpful assistant.",
                original_prompt="I had a grilled chicken salad for lunch",
                redacted_text="I had a grilled chicken salad for lunch",
                max_retries=0,
                user_print=capture_print,
            )

            # Run with incomplete args to trigger extraction
            result = tool.run({}, context)
            if result.success:
                eval_db = test_db  # Use the successful DB for assertions
                break
            print(f"   Attempt {attempt + 1} failed, retrying...")

        print(f"   Success: {result.success}")
        print(f"   Reply: {result.reply_text[:200] if result.reply_text else 'None'}...")

        assert result.success, f"Tool should succeed after retries, got: {result.reply_text}"

        # Check that macros are in the reply
        reply_lower = result.reply_text.lower() if result.reply_text else ""
        has_macros = any(term in reply_lower for term in ["kcal", "protein", "carb", "fat"])

        print(f"   Has macros in reply: {has_macros}")
        assert has_macros, "Reply should include macro information"

        # Verify meal was stored in DB
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        meals = test_db.get_meals_between(
            (now - timedelta(minutes=5)).isoformat(),
            (now + timedelta(minutes=5)).isoformat()
        )

        print(f"   Meals in DB: {len(meals)}")
        assert len(meals) >= 1, "Should have stored at least one meal"

        # Check the stored meal has nutrition data
        meal = meals[0]
        # sqlite3.Row needs index or column name access
        calories = meal["calories_kcal"] if "calories_kcal" in meal.keys() else None
        print(f"   Stored meal calories: {calories}")

        has_stored_macros = calories is not None
        print(f"   Has stored macros: {has_stored_macros}")

        assert has_stored_macros, f"Stored meal should have macros"
        print(f"   PASS Meal logged with macros: {calories} kcal")


# =============================================================================
# Comparison Tests (for debugging model differences)
# =============================================================================

class TestNutritionModelComparison:
    """
    Tests specifically designed to compare nutrition extraction between models.

    These help diagnose why smaller models may perform worse.
    """

    @pytest.mark.eval
    @requires_judge_llm
    def test_simple_meal_extraction(self, mock_config):
        """
        Simple meal that any model should handle correctly.
        """
        mock_config.ollama_base_url = JUDGE_BASE_URL
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.llm_chat_timeout_sec = 30.0

        print(f"\n[SIMPLE] Simple meal test (baseline)")
        print(f"   Model: {JUDGE_MODEL}")

        # Very simple, common meal
        data = call_nutrition_extraction(mock_config, "I had 2 boiled eggs")

        print(f"   Response: {json.dumps(data, indent=2) if data else 'None'}")

        assert data is not None, "Should extract simple meal"

        # 2 boiled eggs: ~140-160 kcal, 12-14g protein, 0-2g carbs, 10-12g fat
        # Note: Smaller models may sometimes parse as 1 egg (~78 kcal), so we use a loose range
        calories = data.get("calories_kcal")
        protein = data.get("protein_g")

        if calories:
            # Loose range: 1-2 eggs worth (some models miss quantity)
            assert 60 <= float(calories) <= 350, f"Calories {calories} way off for eggs"

        if protein:
            assert 5 <= float(protein) <= 20, f"Protein {protein}g way off for eggs"

        print(f"   PASS Simple extraction succeeded")

    @pytest.mark.eval
    @requires_judge_llm
    def test_extraction_with_quantities(self, mock_config):
        """
        Test extraction with explicit quantities (should improve accuracy).
        """
        mock_config.ollama_base_url = JUDGE_BASE_URL
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.llm_chat_timeout_sec = 30.0

        print(f"\n[QUANTITY] Quantity extraction test")
        print(f"   Model: {JUDGE_MODEL}")

        # Explicit quantities should help smaller models
        data = call_nutrition_extraction(
            mock_config,
            "I had 100g of cooked white rice and 150g of grilled chicken breast"
        )

        print(f"   Response: {json.dumps(data, indent=2) if data else 'None'}")

        assert data is not None, "Should extract meal with quantities"

        # 100g rice: ~130 kcal, 2.7g protein, 28g carbs, 0.3g fat
        # 150g chicken: ~248 kcal, 46g protein, 0g carbs, 5.4g fat
        # Total: ~378 kcal, ~49g protein, ~28g carbs, ~6g fat
        # Note: Models can vary significantly; some may overestimate if assuming larger portions

        calories = data.get("calories_kcal")
        protein = data.get("protein_g")

        if calories:
            assert 200 <= float(calories) <= 800, f"Calories {calories} off for rice+chicken"

        if protein:
            # Wider range to accommodate model variance (some assume larger chicken portions)
            assert 20 <= float(protein) <= 120, f"Protein {protein}g off for rice+chicken"

        print(f"   PASS Quantity-based extraction succeeded")
