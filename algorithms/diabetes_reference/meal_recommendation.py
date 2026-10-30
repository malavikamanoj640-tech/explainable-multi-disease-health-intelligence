# ============================================================
# DualCare - Meal Recommendation System
# Personalized Meal Plan Generator
# ============================================================


def generate_meal_plan(
    risk_level,
    diet="Vegetarian",
    cuisine="Indian",
    activity="Moderate"
):

    # ========================================================
    # NORMALIZE INPUTS
    # ========================================================

    risk_level = str(
        risk_level
    ).strip()

    diet = str(
        diet
    ).strip()

    cuisine = str(
        cuisine
    ).strip()

    activity = str(
        activity
    ).strip()


    # ========================================================
    # SUPPORTED DIETS
    # ========================================================

    supported_diets = [

        "Vegetarian",

        "Non-Vegetarian",

        "Eggetarian",

        "Vegan"
    ]


    # --------------------------------------------------------
    # Default to Vegetarian if invalid diet is supplied
    # --------------------------------------------------------

    if diet not in supported_diets:

        diet = "Vegetarian"


    # ========================================================
    # SUPPORTED CUISINES
    # ========================================================

    supported_cuisines = [

        "Indian"
    ]


    # --------------------------------------------------------
    # Default cuisine
    # --------------------------------------------------------

    if cuisine not in supported_cuisines:

        cuisine = "Indian"


    # ========================================================
    # SUPPORTED ACTIVITY LEVELS
    # ========================================================

    supported_activity = [

        "Low",

        "Moderate",

        "High"
    ]


    if activity not in supported_activity:

        activity = "Moderate"


    # ========================================================
    # BASE MEAL OPTIONS
    # ========================================================

    indian_vegetarian = {

        "breakfast": [

            "Vegetable oats with minimal added salt",

            "Moong dal chilla with vegetables",

            "Vegetable poha with a small portion of "
            "unsweetened or low-sugar curd"
        ],

        "lunch": [

            "2 small whole-wheat chapatis",

            "One bowl dal",

            "Mixed non-starchy vegetables",

            "Fresh salad"
        ],

        "snack": [

            "One small apple or guava",

            "A small portion of unsalted nuts"
        ],

        "dinner": [

            "Vegetable soup",

            "1–2 small whole-wheat chapatis",

            "Mixed vegetables",

            "Small portion of paneer or dal"
        ]
    }


    # ========================================================
    # NON-VEGETARIAN
    # ========================================================

    indian_nonveg = {

        "breakfast": [

            "Vegetable oats",

            "One boiled egg",

            "Unsweetened tea or coffee"
        ],

        "lunch": [

            "2 small whole-wheat chapatis",

            "Grilled or lightly cooked chicken",

            "Mixed non-starchy vegetables",

            "Fresh salad"
        ],

        "snack": [

            "One small apple or guava",

            "A small portion of unsalted nuts"
        ],

        "dinner": [

            "Vegetable soup",

            "Grilled chicken or fish",

            "Mixed vegetables",

            "Small portion of whole grains"
        ]
    }


    # ========================================================
    # EGGETARIAN
    # ========================================================

    indian_eggetarian = {

        "breakfast": [

            "Vegetable oats",

            "One or two boiled eggs",

            "Unsweetened tea or coffee"
        ],

        "lunch": [

            "2 small whole-wheat chapatis",

            "Dal",

            "Egg curry prepared with minimal oil",

            "Mixed vegetables",

            "Fresh salad"
        ],

        "snack": [

            "One small apple or guava",

            "A small portion of unsalted nuts"
        ],

        "dinner": [

            "Vegetable soup",

            "Vegetable omelette or boiled eggs",

            "Mixed vegetables",

            "Small portion of whole grains"
        ]
    }


    # ========================================================
    # VEGAN
    # ========================================================

    vegan_meals = {

        "breakfast": [

            "Vegetable oats prepared without dairy",

            "Moong dal chilla",

            "One small portion of whole fruit"
        ],

        "lunch": [

            "2 small whole-wheat chapatis",

            "Dal",

            "Mixed vegetables",

            "Fresh salad"
        ],

        "snack": [

            "Apple or guava",

            "Small portion of unsalted nuts or seeds"
        ],

        "dinner": [

            "Vegetable soup",

            "Tofu with mixed vegetables",

            "Small portion of whole grains"
        ]
    }


    # ========================================================
    # SELECT DIET
    # ========================================================

    if diet == "Non-Vegetarian":

        meal_plan = {

            key: value.copy()

            for key, value
            in indian_nonveg.items()
        }


    elif diet == "Eggetarian":

        meal_plan = {

            key: value.copy()

            for key, value
            in indian_eggetarian.items()
        }


    elif diet == "Vegan":

        meal_plan = {

            key: value.copy()

            for key, value
            in vegan_meals.items()
        }


    else:

        meal_plan = {

            key: value.copy()

            for key, value
            in indian_vegetarian.items()
        }


    # ========================================================
    # PREFERRED FOODS
    # ========================================================

    prefer = [

        "Non-starchy vegetables",

        "Whole grains in appropriate portions",

        "Pulses and legumes",

        "High-fiber foods",

        "Unsalted nuts and seeds",

        "Whole fruits in appropriate portions",

        "Water and unsweetened beverages"
    ]


    # ========================================================
    # DIET-SPECIFIC PREFERENCES
    # ========================================================

    if diet == "Vegetarian":

        prefer.extend([

            "Dal, beans and other plant-based protein sources",

            "Low-fat or moderate portions of dairy products",

            "Paneer in appropriate portions"
        ])


    elif diet == "Non-Vegetarian":

        prefer.extend([

            "Lean chicken or fish prepared with minimal oil",

            "Eggs in appropriate portions",

            "Lean protein instead of processed meat"
        ])


    elif diet == "Eggetarian":

        prefer.extend([

            "Eggs prepared with minimal oil",

            "Dal, beans and other plant-based protein sources",

            "Vegetables combined with adequate protein"
        ])


    elif diet == "Vegan":

        prefer.extend([

            "Tofu and other plant-based protein sources",

            "Beans, lentils and chickpeas",

            "Nuts and seeds in appropriate portions",

            "Unsweetened plant-based foods"
        ])


    # ========================================================
    # FOODS TO LIMIT
    # ========================================================

    limit = [

        "Sugar-sweetened beverages",

        "Sweets and desserts",

        "Highly refined carbohydrate foods",

        "Highly processed foods",

        "Excessive portions",

        "Foods high in added sugar",

        "Deep-fried foods",

        "Excessive added salt"
    ]


    # ========================================================
    # DIET-SPECIFIC FOODS TO LIMIT
    # ========================================================

    if diet == "Non-Vegetarian":

        limit.extend([

            "Processed meats such as sausages and salami",

            "Deep-fried meat and fish",

            "High-fat meat preparations"
        ])


    elif diet == "Eggetarian":

        limit.extend([

            "Deep-fried egg preparations",

            "Egg dishes prepared with excessive oil"
        ])


    elif diet == "Vegetarian":

        limit.extend([

            "Deep-fried paneer",

            "High-sugar dairy products"
        ])


    elif diet == "Vegan":

        limit.extend([

            "Highly processed plant-based foods",

            "Sweetened plant-based beverages"
        ])


    # ========================================================
    # HIGH-RISK ADJUSTMENT
    # ========================================================
    #
    # IMPORTANT:
    # The selected diet is preserved.
    # ========================================================

    if risk_level == "High Risk":

        # ----------------------------------------------------
        # Vegetarian
        # ----------------------------------------------------

        if diet == "Vegetarian":

            meal_plan["breakfast"] = [

                "Vegetable oats with minimal added sugar",

                "Moong dal chilla with vegetables",

                "Unsweetened tea or coffee"
            ]

            meal_plan["lunch"] = [

                "2 small whole-wheat chapatis",

                "Dal",

                "Large portion of non-starchy vegetables",

                "Fresh salad"
            ]

            meal_plan["snack"] = [

                "Small portion of unsalted nuts",

                "One small whole fruit"
            ]

            meal_plan["dinner"] = [

                "Vegetable soup",

                "Mixed vegetables",

                "Small portion of dal or paneer",

                "Small portion of whole grains"
            ]


        # ----------------------------------------------------
        # Non-Vegetarian
        # ----------------------------------------------------

        elif diet == "Non-Vegetarian":

            meal_plan["breakfast"] = [

                "Vegetable oats",

                "One boiled egg",

                "Unsweetened tea or coffee"
            ]

            meal_plan["lunch"] = [

                "2 small whole-wheat chapatis",

                "Grilled chicken or fish",

                "Large portion of non-starchy vegetables",

                "Fresh salad"
            ]

            meal_plan["snack"] = [

                "Small portion of unsalted nuts",

                "One small whole fruit"
            ]

            meal_plan["dinner"] = [

                "Vegetable soup",

                "Grilled fish or skinless chicken",

                "Mixed vegetables",

                "Small portion of whole grains"
            ]


        # ----------------------------------------------------
        # Eggetarian
        # ----------------------------------------------------

        elif diet == "Eggetarian":

            meal_plan["breakfast"] = [

                "Vegetable oats",

                "One or two boiled eggs",

                "Unsweetened tea or coffee"
            ]

            meal_plan["lunch"] = [

                "2 small whole-wheat chapatis",

                "Dal",

                "Egg curry prepared with minimal oil",

                "Large portion of non-starchy vegetables",

                "Fresh salad"
            ]

            meal_plan["snack"] = [

                "Small portion of unsalted nuts",

                "One small whole fruit"
            ]

            meal_plan["dinner"] = [

                "Vegetable soup",

                "Boiled eggs or vegetable omelette",

                "Mixed vegetables",

                "Small portion of whole grains"
            ]


        # ----------------------------------------------------
        # Vegan
        # ----------------------------------------------------

        elif diet == "Vegan":

            meal_plan["breakfast"] = [

                "Vegetable oats prepared without dairy",

                "Moong dal chilla",

                "Unsweetened plant-based beverage"
            ]

            meal_plan["lunch"] = [

                "2 small whole-wheat chapatis",

                "Dal or chickpeas",

                "Large portion of non-starchy vegetables",

                "Fresh salad"
            ]

            meal_plan["snack"] = [

                "Small portion of unsalted nuts or seeds",

                "One small whole fruit"
            ]

            meal_plan["dinner"] = [

                "Vegetable soup",

                "Tofu with mixed vegetables",

                "Small portion of whole grains"
            ]


        # ----------------------------------------------------
        # High-risk guidance
        # ----------------------------------------------------

        prefer.extend([

            "Prefer meals with vegetables and adequate protein",

            "Choose high-fiber carbohydrate sources",

            "Keep portions of refined carbohydrates controlled",

            "Prefer water and unsweetened beverages"
        ])

        limit.extend([

            "Large portions of refined carbohydrates",

            "Sugar-sweetened drinks",

            "Frequent desserts and sweets"
        ])


    # ========================================================
    # MODERATE-RISK ADJUSTMENT
    # ========================================================

    elif risk_level == "Moderate Risk":

        prefer.extend([

            "Choose balanced meals containing vegetables, "
            "protein and high-fiber carbohydrates",

            "Keep meal portions consistent",

            "Prefer minimally processed foods"
        ])

        limit.extend([

            "Frequent sugary snacks",

            "Large portions of refined carbohydrates"
        ])


    # ========================================================
    # LOW-RISK ADJUSTMENT
    # ========================================================

    elif risk_level == "Low Risk":

        prefer.extend([

            "Continue balanced eating habits",

            "Include vegetables and fiber-rich foods regularly",

            "Maintain appropriate meal portions"
        ])


    # ========================================================
    # ACTIVITY LEVEL ADJUSTMENT
    # ========================================================

    if activity == "Low":

        prefer.append(

            "Choose appropriately sized portions "
            "and include regular light physical activity "
            "as appropriate for your health."
        )

        limit.append(

            "Frequent high-calorie snacks "
            "and oversized portions"
        )


    elif activity == "Moderate":

        prefer.append(

            "Maintain balanced meals with adequate "
            "protein, fiber and vegetables."
        )


    elif activity == "High":

        prefer.extend([

            "Balanced meals with adequate protein and fiber",

            "Include nutritious foods around periods "
            "of higher physical activity",

            "Maintain adequate hydration"
        ])


    # ========================================================
    # GENERAL HEALTH GUIDANCE
    # ========================================================

    tips = [

        "Drink adequate water throughout the day.",

        "Prefer whole or minimally processed foods.",

        "Limit added sugar and highly processed foods.",

        "Include vegetables and fiber-rich foods regularly.",

        "Use moderate portions of whole grains.",

        "Choose unsweetened beverages when possible."
    ]


    # ========================================================
    # RISK-SPECIFIC TIPS
    # ========================================================

    if risk_level == "High Risk":

        tips.extend([

            "Because the model estimates higher predicted "
            "risk, consider discussing your health status "
            "with a qualified healthcare professional.",

            "Do not use this meal plan as a substitute "
            "for professional medical or nutritional advice."
        ])


    elif risk_level == "Moderate Risk":

        tips.extend([

            "Continue preventive lifestyle habits "
            "and appropriate health monitoring."
        ])


    else:

        tips.extend([

            "Continue maintaining healthy lifestyle habits "
            "and appropriate preventive care."
        ])


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    prefer = list(

        dict.fromkeys(
            prefer
        )
    )


    limit = list(

        dict.fromkeys(
            limit
        )
    )


    tips = list(

        dict.fromkeys(
            tips
        )
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    meal_plan["prefer"] = prefer

    meal_plan["limit"] = limit

    meal_plan["tips"] = tips

    meal_plan["diet"] = diet

    meal_plan["cuisine"] = cuisine

    meal_plan["activity"] = activity

    meal_plan["risk_level"] = risk_level


    return meal_plan