import os
import requests
import gradio as gr

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# Función que realiza request al backend
def predict(
    X, Y, size,
    num_deliver_per_week, num_visit_per_week,
    mean_past_items, mean_purchases_per_week, mean_sales_per_week,
    weeks_since_last_purchase,
    brand, sub_category, segment, package, customer_type
):
    payload = {
        "X": X,
        "Y": Y,
        "size": size,
        "num_deliver_per_week": num_deliver_per_week,
        "num_visit_per_week": num_visit_per_week,
        "mean_past_items": mean_past_items,
        "mean_purchases_per_week": mean_purchases_per_week,
        "mean_sales_per_week": mean_sales_per_week,
        "weeks_since_last_purchase": weeks_since_last_purchase,
        "brand": brand,
        "sub_category": sub_category,
        "segment": segment,
        "package": package,
        "customer_type": customer_type,
    }
    r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
    resultado = int(r.json()["prediction"])
    if resultado == 1:
        return "El cliente comprará la próxima semana"
    else:
        return "El cliente no comprará la próxima semana"

# Interfaz con gradio
with gr.Blocks(title="SodAI Drinks - Predictor") as interface:
    gr.Markdown("# Predictor de compra de SodAI Drinks")
    gr.Markdown("## Bienvenido! Aquí puedes predecir si un cliente comprará una bebida la próxima semana")
    gr.Markdown("### Ingresa las características del cliente")
    with gr.Row():
        X      = gr.Number(label="Coordenada X del cliente")
        Y      = gr.Number(label="Coordenada Y del cliente")
        customer_type = gr.Dropdown(["ABARROTES", "MAYORISTA", "MINIMARKET", "CANAL FRIO", "RESTAURANT", "SUPERMERCADO", "TIENDA DE CONVENIENCIA"], label="Tipo de cliente")

    with gr.Row():
        num_deliver_per_week  = gr.Number(label="# entregas/semana")
        num_visit_per_week    = gr.Number(label="# visitas/semana")
        mean_purchases_per_week  = gr.Number(label="Volumen promedio semanal que compra el cliente")
    
    gr.Markdown("### Ingresa las características de la bebida")
    with gr.Row():
        brand         = gr.Textbox(label="Marca")
        size   = gr.Number(label="Tamaño (L)")
        mean_sales_per_week      = gr.Number(label="Promedio de venta semanal por cliente")

    with gr.Row():
        sub_category  = gr.Dropdown(["GASEOSAS", "JUGOS", "AGUAS SABORIZADAS"], label="Sub-categoría")
        segment       = gr.Dropdown(["LOW", "PREMIUM", "MEDIUM", "HIGH"], label="Segmento")
        package       = gr.Dropdown(["BOTELLA", "LATA", "TETRA", "KEG"], label="Envase")

    gr.Markdown("### Ingresa el comportamiento reciente del cliente con esta bebida")

    with gr.Row():
        mean_past_items          = gr.Number(label="Promedio de venta semanal a este cliente")
        weeks_since_last_purchase = gr.Number(label="Semanas desde la última venta a este cliente")

    btn = gr.Button("Predecir")
    out = gr.Textbox(label="Resultado")

    btn.click(
        predict,
        inputs=[
            X, Y, size,
            num_deliver_per_week, num_visit_per_week,
            mean_past_items, mean_purchases_per_week, mean_sales_per_week,
            weeks_since_last_purchase,
            brand, sub_category, segment, package, customer_type,
        ],
        outputs=out
    )

interface.launch(server_name="0.0.0.0", server_port=7860)