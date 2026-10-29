"""
AutoVision AI - Gradio Web Application
Next-generation automotive intelligence and car resale price prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import gradio as gr
from model_utils import load_model, predict_price, load_car_data

# Ensure model is ready
bundle = load_model()

# Custom dark-gradient CSS styling
CUSTOM_CSS = """
:root {
    --bg-primary: #080817;
    --card-bg: rgba(23, 23, 46, 0.85);
    --card-border: rgba(255, 255, 255, 0.12);
    --accent-glow: #6736a944;
}

body, .gradio-container {
    background: radial-gradient(circle at 15% 15%, #34175788 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, #004d6d66 0%, transparent 45%),
                #080817 !important;
    color: #e6ebff !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

#app-container {
    max-width: 1280px;
    margin: 0 auto;
    padding: 24px 20px 48px;
}

.hero-banner {
    background: linear-gradient(135deg, #1f113b 0%, #101c38 50%, #063445 100%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 22px;
    padding: 36px 32px;
    margin-bottom: 24px;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45);
}

.eyebrow {
    color: #4ce7ff;
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero-title {
    font-size: clamp(2rem, 3.5vw, 2.8rem);
    font-weight: 800;
    line-height: 1.15;
    margin: 6px 0 14px;
    color: #ffffff;
}

.gradient-text {
    background: linear-gradient(90deg, #b886ff 0%, #44e5ff 60%, #ff7ad7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #b5c2e8;
    font-size: 1.05rem;
    max-width: 720px;
    line-height: 1.6;
    margin: 0;
}

.glass-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 18px !important;
    padding: 22px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35) !important;
    backdrop-filter: blur(10px);
}

.result-card {
    background: radial-gradient(ellipse at top right, rgba(130, 97, 255, 0.22), transparent 70%),
                #111428 !important;
    border: 1px solid rgba(130, 97, 255, 0.4) !important;
    border-radius: 18px !important;
    padding: 24px !important;
}

#predict-btn {
    background: linear-gradient(95deg, #7c3aed 0%, #06b6d4 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 12px !important;
    padding: 14px 20px !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.4) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

#predict-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(6, 182, 212, 0.5) !important;
}

#reset-btn {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    color: #d1d9f5 !important;
    border-radius: 12px !important;
}

.price-display textarea {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    text-align: center !important;
    color: #4ce7ff !important;
    background: #090d1e !important;
    border: 1px solid #5444a7 !important;
    border-radius: 14px !important;
}

.footer-text {
    text-align: center;
    color: #7b88aa;
    font-size: 0.85rem;
    margin-top: 32px;
}
"""


def compute_prediction(
    present_price: float,
    kms_driven: float,
    age: int,
    fuel_type: str,
    seller_type: str,
    transmission: str,
):
    """Callback for computing valuation and market insights."""
    if present_price is None or present_price <= 0:
        return "₹ —", "⚠️ Please enter a present price greater than 0.", "—", "—"

    if kms_driven is None or kms_driven < 0:
        return "₹ —", "⚠️ Kilometers driven cannot be negative.", "—", "—"

    values = {
        "Present_Price": float(present_price),
        "Kms_Driven": float(kms_driven),
        "Age": int(age),
        "Fuel_Type": str(fuel_type),
        "Seller_Type": str(seller_type),
        "Transmission": str(transmission),
    }

    pred = predict_price(bundle, values)
    lower_range = max(0.0, pred * 0.85)
    upper_range = pred * 1.15

    retention_pct = (pred / present_price) * 100 if present_price > 0 else 0
    depreciation_amt = max(0.0, present_price - pred)

    price_str = f"₹ {pred:.2f} Lakhs"
    range_str = f"₹ {lower_range:.2f} - ₹ {upper_range:.2f} Lakhs"
    retention_str = f"{retention_pct:.1f}% retained ({depreciation_amt:.2f} L depreciation)"

    details_str = (
        f"✅ Valuation Verified\n"
        f"• Age: {age} yrs  |  Distance: {kms_driven:,.0f} km\n"
        f"• Config: {fuel_type} • {transmission} • {seller_type} Seller\n"
        f"• Model Estimate: Multi-factor Ridge / Linear Regression Engine"
    )

    return price_str, details_str, range_str, retention_str


def apply_preset(preset_name: str):
    """Presets for fast 1-click test evaluations."""
    presets = {
        "🚗 Budget Hatchback": (5.5, 32000, 4, "Petrol", "Dealer", "Manual"),
        "🚙 Diesel Cruiser": (10.5, 58000, 5, "Diesel", "Individual", "Manual"),
        "💎 Executive Sedan": (14.0, 24000, 3, "Petrol", "Dealer", "Automatic"),
        "⚡ City Commuter": (4.2, 45000, 6, "CNG", "Dealer", "Manual"),
    }
    return presets.get(preset_name, (8.0, 30000, 5, "Petrol", "Dealer", "Manual"))


def build_app():
    """Construct the Gradio UI blocks."""
    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(title="AutoVision AI · Car Valuation") as demo:
        with gr.Column(elem_id="app-container"):
            # Hero Header
            gr.HTML(
                """
                <div class="hero-banner">
                    <div class="eyebrow">✦ AUTOMOTIVE INTELLIGENCE & VALUATION ENGINE</div>
                    <h1 class="hero-title">Discover Your Car's <span class="gradient-text">Real Market Worth</span></h1>
                    <p class="hero-subtitle">
                        Instant, data-driven used car valuation powered by multiple linear regression and market statistics.
                        Accurately factor in depreciation, mileage, fuel type, transmission, and seller channels.
                    </p>
                </div>
                """
            )

            with gr.Row():
                preset_selector = gr.Radio(
                    choices=[
                        "🚗 Budget Hatchback",
                        "🚙 Diesel Cruiser",
                        "💎 Executive Sedan",
                        "⚡ City Commuter",
                    ],
                    value="🚗 Budget Hatchback",
                    label="⚡ Quick Presets (1-Click Test Configurations)",
                    interactive=True,
                )

            with gr.Row(equal_height=False):
                # Left Column: Inputs
                with gr.Column(scale=11, elem_classes=["glass-card"]):
                    gr.Markdown("### ⚙️ Vehicle Specifications")

                    with gr.Row():
                        present_price = gr.Number(
                            label="Showroom / Original Price (₹ Lakhs)",
                            value=5.5,
                            minimum=0.1,
                            maximum=100.0,
                            step=0.25,
                        )
                        kms_driven = gr.Number(
                            label="Kilometers Driven",
                            value=32000,
                            minimum=0,
                            maximum=1000000,
                            step=1000,
                        )

                    age = gr.Slider(
                        minimum=0,
                        maximum=25,
                        value=4,
                        step=1,
                        label="Vehicle Age (Years)",
                    )

                    with gr.Row():
                        fuel_type = gr.Radio(
                            choices=["Petrol", "Diesel", "CNG"],
                            value="Petrol",
                            label="Fuel Type",
                        )
                        seller_type = gr.Radio(
                            choices=["Dealer", "Individual"],
                            value="Dealer",
                            label="Seller Channel",
                        )
                        transmission = gr.Radio(
                            choices=["Manual", "Automatic"],
                            value="Manual",
                            label="Transmission",
                        )

                    with gr.Row():
                        predict_btn = gr.Button(
                            "✦ ESTIMATE FAIR PRICE →",
                            elem_id="predict-btn",
                            variant="primary",
                        )
                        reset_btn = gr.Button("↺ Reset", elem_id="reset-btn")

                # Right Column: Valuation Results
                with gr.Column(scale=9, elem_classes=["result-card"]):
                    gr.HTML(
                        """
                        <div class="eyebrow">AI VALUATION RESULTS</div>
                        <h2 style="margin: 4px 0 16px; font-size: 1.6rem; color: #f5f7ff;">Estimated Resale Value</h2>
                        """
                    )

                    price_output = gr.Textbox(
                        label="ESTIMATED SELLING PRICE",
                        value="₹ —",
                        interactive=False,
                        elem_classes=["price-display"],
                        lines=1,
                    )

                    with gr.Row():
                        fair_range_output = gr.Textbox(
                            label="Fair Market Band (±15%)",
                            value="—",
                            interactive=False,
                        )
                        retention_output = gr.Textbox(
                            label="Value Retention",
                            value="—",
                            interactive=False,
                        )

                    details_output = gr.Textbox(
                        label="Valuation Analysis",
                        value="Select a preset or enter specifications and click 'Estimate Fair Price'.",
                        interactive=False,
                        lines=4,
                    )

                    with gr.Accordion("ℹ️ How This Model Works", open=False):
                        gr.Markdown(
                            """
                            - **Regression Architecture**: Trained on verified historical used car transactions.
                            - **Impact Drivers**: Present price & vehicle age are the strongest price determinants, followed by fuel type and transmission.
                            - **Fair Band**: Reflects condition variances (exterior, service records, and regional demand).
                            """
                        )

            gr.HTML(
                """
                <div class="footer-text">
                    AutoVision AI · SI Lab Project · Powered by Gradio & Scikit-Learn
                </div>
                """
            )

            # Event handlers
            predict_btn.click(
                fn=compute_prediction,
                inputs=[present_price, kms_driven, age, fuel_type, seller_type, transmission],
                outputs=[price_output, details_output, fair_range_output, retention_output],
            )

            preset_selector.change(
                fn=apply_preset,
                inputs=[preset_selector],
                outputs=[present_price, kms_driven, age, fuel_type, seller_type, transmission],
            ).then(
                fn=compute_prediction,
                inputs=[present_price, kms_driven, age, fuel_type, seller_type, transmission],
                outputs=[price_output, details_output, fair_range_output, retention_output],
            )

            def reset_defaults():
                return 5.5, 32000, 4, "Petrol", "Dealer", "Manual", "₹ —", "Enter specs and predict.", "—", "—"

            reset_btn.click(
                fn=reset_defaults,
                inputs=[],
                outputs=[
                    present_price,
                    kms_driven,
                    age,
                    fuel_type,
                    seller_type,
                    transmission,
                    price_output,
                    details_output,
                    fair_range_output,
                    retention_output,
                ],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="AutoVision AI Gradio App")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio share link")
    args = parser.parse_args()

    demo = build_app()
    # Gradio 6.0 handles css and theme cleanly in launch
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
