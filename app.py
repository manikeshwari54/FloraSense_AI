"""
FloraSense AI — Indoor Plant Identification & Care Assistant
-------------------------------------------------------------
Identifies 20 common indoor plant species using transfer learning CNNs.
Dataset: kakasher/house-plant-species (CC BY 3.0, 14,790 images, 47 classes)
"""

import io
import json
from html import escape
from pathlib import Path
import streamlit as st
from PIL import Image

from dataset import SPECIES_TAXONOMY
from model import InferenceEngine, load_inference_engine
from ood_detector import check_image_quality, calibrate_and_filter, load_thresholds
from plant_data import get_plant_profile

st.set_page_config(
    page_title="FloraSense | Plant Intelligence",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="auto"
)

styles_path = Path(__file__).with_name("styles.css")
if styles_path.exists():
    st.markdown(f"<style>{styles_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner='Loading EfficientNet-B0...')
def get_efficientnet_engine():
    engine, err = load_inference_engine('./checkpoints', 'efficientnet_b0')
    return engine, err

@st.cache_resource(show_spinner='Loading MobileNetV3-Small...')
def get_mobilenet_engine():
    engine, err = load_inference_engine('./checkpoints', 'mobilenet_v3_small')
    return engine, err

PAGES = ("Home", "Identify Plant", "Plant Care", "Supported Plants", "Model Information", "About Project")
CLASS_MAPPING_PATH = Path(__file__).with_name("checkpoints") / "class_mapping.json"
if CLASS_MAPPING_PATH.exists():
    with CLASS_MAPPING_PATH.open("r", encoding="utf-8") as mapping_file:
        active_mapping = json.load(mapping_file).get("id2label", {})
    SUPPORTED_CLASSES = [
        active_mapping[key]
        for key in sorted(active_mapping, key=int)
    ]
else:
    SUPPORTED_CLASSES = []


def reset_identification():
    st.session_state["last_result"] = None
    st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1


def navigate_to_identification():
    st.session_state["navigation"] = "Identify Plant"


def show_profile_field(profile, key, label):
    value = profile.get(key)
    with st.container(border=True):
        st.markdown(f"<div class='field-label'>{escape(label)}</div>", unsafe_allow_html=True)
        if value:
            st.write(value)
        else:
            st.caption("Information not available in the care database.")


def render_care_profile(profile):
    st.markdown(f"## {escape(profile.get('common_name', 'Plant care'))}")
    st.markdown(f"<p class='scientific-name'>{escape(profile.get('scientific_name', ''))}</p>", unsafe_allow_html=True)
    st.write(profile.get("description") or "Description not available in the care database.")

    overview, growing, environment, maintenance, safety = st.tabs([
        "Overview", "Light & Water", "Environment", "Maintenance", "Pet Safety"
    ])
    with overview:
        left, right = st.columns(2)
        with left:
            show_profile_field(profile, "plant_type", "Plant type")
            show_profile_field(profile, "family", "Family")
        with right:
            show_profile_field(profile, "maintenance", "Maintenance level")
            show_profile_field(profile, "location", "Suggested location")
        if profile.get("features"):
            st.markdown("#### Plant characteristics")
            for feature in profile["features"]:
                st.markdown(f"- {feature}")
    with growing:
        left, right = st.columns(2)
        with left:
            show_profile_field(profile, "sunlight", "Sunlight")
            show_profile_field(profile, "water", "Watering")
        with right:
            show_profile_field(profile, "soil", "Soil")
    with environment:
        show_profile_field(profile, "temperature_humidity", "Temperature & humidity")
    with maintenance:
        show_profile_field(profile, "pruning_care", "Maintenance guidance")
    with safety:
        safety_text = profile.get("toxicity_safety")
        if not safety_text:
            st.info("Pet-safety information is not available in the care database.")
        elif "non-toxic" in safety_text.lower() or "safe" in safety_text.lower():
            st.markdown(
                f"<div class='safety-safe'><strong>Safety information</strong><br>{escape(safety_text)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='safety-alert'><strong>Safety warning</strong><br>{escape(safety_text)}</div>",
                unsafe_allow_html=True,
            )
        st.caption("Safety guidance is informational. Confirm with a veterinarian or poison-control resource when needed.")


def render_result(result):
    if not result:
        with st.container(border=True):
            st.markdown("### Your identification will appear here")
            st.caption("Upload a clear photo of one plant, then select Identify.")
        return

    predictions = result.get("predictions", [])
    if result.get("status") == "unsuitable":
        st.error(result.get("message", "This image could not be processed."))
        return
    if not predictions:
        st.info("No prediction is available for this image.")
        return

    top_prediction = predictions[0]
    if result.get("status") == "uncertain":
        st.warning("Identification is uncertain. Treat these as unconfirmed candidates, not a confirmed result.")
        st.markdown(
            f"## {escape(top_prediction.get('common_name', 'Unconfirmed candidate'))}"
            f"\n\n<p class='scientific-name'>{escape(top_prediction.get('scientific_name', ''))}</p>",
            unsafe_allow_html=True,
        )
    else:
        with st.container(border=True):
            st.markdown(f"## {escape(top_prediction.get('common_name', 'Plant identified'))}")
            st.markdown(
                f"<p class='scientific-name'>{escape(top_prediction.get('scientific_name', ''))}</p>",
                unsafe_allow_html=True,
            )
            st.metric("Model confidence", f"{top_prediction.get('score_pct', 0)}%")
            st.caption("Confidence is the model's score, not a guarantee of identification accuracy.")

    if len(predictions) > 1:
        st.markdown("#### Alternative candidates" if result.get("status") != "uncertain" else "#### Unconfirmed candidates")
        for prediction in predictions[1:3] if result.get("status") != "uncertain" else predictions[:3]:
            label_col, score_col = st.columns([4, 1])
            with label_col:
                st.write(f"**{prediction.get('common_name', 'Unknown')}**")
                st.caption(prediction.get("scientific_name", ""))
            with score_col:
                st.write(f"{prediction.get('score_pct', 0)}%")

    profile = get_plant_profile(top_prediction.get("common_name", ""))
    if profile and result.get("status") == "uncertain":
        with st.expander("Care reference for this unconfirmed candidate"):
            st.info("The identification is unconfirmed. Use this profile as a reference only.")
            show_profile_field(profile, "sunlight", "Sunlight")
            show_profile_field(profile, "water", "Watering")
            show_profile_field(profile, "toxicity_safety", "Pet safety")
    elif profile and result.get("status") == "certain":
        st.markdown("### Care profile")
        render_care_profile(profile)


with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'><span class='brand-mark'>F</span>"
        "<span><strong>FloraSense</strong><small>Plant intelligence</small></span></div>",
        unsafe_allow_html=True,
    )
    page = st.radio("Navigate", PAGES, key="navigation", label_visibility="collapsed")
    st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)
    st.markdown("#### Inference settings")
    model_choice = st.selectbox(
        "Model architecture",
        ["EfficientNet-B0 (Primary)", "MobileNetV3-Small (Lightweight)"],
        index=0,
    )
    default_thresholds = load_thresholds("./checkpoints/ood_thresholds.json")
    with st.expander("Uncertainty filtering", expanded=False):
        enable_ood = st.checkbox("Enable OOD filtering", value=True)
        energy_thresh = st.slider(
            "Energy threshold",
            min_value=-40.0,
            max_value=-5.0,
            value=float(default_thresholds.get("energy_thresh", -25.0)),
            step=1.0,
            help="Higher values demand stronger evidence before accepting a prediction as certain.",
        )
        conf_thresh = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.95,
            value=float(default_thresholds.get("conf_thresh", 0.35)),
            step=0.05,
            help="Minimum top-1 probability required to confirm a prediction.",
        )
    with st.expander("Checkpoint diagnostics", expanded=False):
        for checkpoint in InferenceEngine.list_available("./checkpoints"):
            st.markdown(f"**{checkpoint['arch'].replace('_', ' ').title()}**")
            if checkpoint["is_usable"]:
                st.caption(f"Ready · {checkpoint['num_classes']} mapped classes")
            else:
                st.caption(checkpoint.get("error") or "Checkpoint unavailable")

st.markdown(
    f"<div class='page-kicker'>BOTANICAL RECOGNITION · LOCAL INFERENCE</div><h1 class='page-title'>{escape(page)}</h1>",
    unsafe_allow_html=True,
)

if page == "Home":
    st.markdown(
        "<section class='welcome-panel'><div class='welcome-copy'>"
        "<p class='eyebrow'>A clearer view of your greenery</p>"
        "<h2>Know your plant.<br>Care for it with confidence.</h2>"
        "<p>FloraSense pairs image classification with a practical botanical care library for common indoor plants.</p>"
        "</div><div class='welcome-visual'><div class='visual-ring'></div><div class='visual-stem'></div>"
        "<div class='visual-leaf leaf-one'></div><div class='visual-leaf leaf-two'></div>"
        "<div class='visual-leaf leaf-three'></div><div class='visual-leaf leaf-four'></div>"
        "<span>FIELD NOTES · 01</span></div></section>",
        unsafe_allow_html=True,
    )
    st.button(
        "Identify a plant",
        type="primary",
        key="home_identify",
        on_click=navigate_to_identification,
    )
    st.markdown("### A practical plant companion")
    feature_columns = st.columns(3)
    features = [
        ("01", "Image identification", "Compare a plant photo against the classes represented in the active model."),
        ("02", "Care library", "Browse sunlight, watering, soil, environment, maintenance, and safety guidance."),
        ("03", "Plant health context", "Use species-specific care notes as a starting point; this tool does not diagnose disease."),
    ]
    for column, (number, title, copy) in zip(feature_columns, features):
        with column:
            with st.container(border=True):
                st.markdown(f"<p class='card-index'>{number}</p><h3>{title}</h3><p>{copy}</p>", unsafe_allow_html=True)
    st.markdown("### In the current model")
    preview_columns = st.columns(4)
    for column, name in zip(preview_columns, SUPPORTED_CLASSES[:4]):
        with column:
            st.markdown(f"<div class='species-chip'>{escape(name)}</div>", unsafe_allow_html=True)
    st.caption(f"The active class mapping contains {len(SUPPORTED_CLASSES)} labels. Browse all supported classes in Supported Plants.")

elif page == "Identify Plant":
    st.markdown("<p class='page-intro'>Add a clear image of one plant. The photo is processed locally by the selected checkpoint.</p>", unsafe_allow_html=True)
    if "upload_generation" not in st.session_state:
        st.session_state["upload_generation"] = 0
    upload_col, result_col = st.columns([1, 1.1], gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown("### Plant photograph")
            uploaded_file = st.file_uploader(
                "Drop an image here or browse files",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"plant_photo_{st.session_state['upload_generation']}",
                help="Supported formats: JPG, JPEG, PNG, WebP.",
            )
            image = None
            if uploaded_file:
                try:
                    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                    st.image(image, caption=f"{image.width} × {image.height} px", use_container_width=True)
                except Exception as error:
                    st.error(f"This image could not be opened: {error}")
            identify_col, reset_col = st.columns(2)
            with identify_col:
                run_classify = st.button("Identify", type="primary", use_container_width=True, disabled=image is None)
            with reset_col:
                st.button("Reset", on_click=reset_identification, use_container_width=True)
            st.caption("Best results usually come from a focused photo with leaves clearly visible.")

    with result_col:
        st.markdown("### Identification")
        if run_classify and image is not None:
            with st.spinner("Checking image quality and running the selected model…"):
                is_valid_quality, quality_err = check_image_quality(image)
                if not is_valid_quality:
                    st.session_state["last_result"] = {
                        "status": "unsuitable",
                        "reason": "image_quality_failed",
                        "message": quality_err,
                        "predictions": [],
                        "model_used": model_choice,
                    }
                else:
                    if "EfficientNet" in model_choice:
                        engine, error = get_efficientnet_engine()
                    else:
                        engine, error = get_mobilenet_engine()
                    if engine is None:
                        st.error(
                            f"Model checkpoint unavailable. {error}\n\n"
                            "The existing model must be trained before this architecture can be used."
                        )
                        st.session_state["last_result"] = None
                    else:
                        logits, predictions = engine.predict(image, top_k=5)
                        if enable_ood:
                            decision = calibrate_and_filter(
                                logits=logits,
                                predictions=predictions,
                                temperature=1.35,
                                energy_thresh=energy_thresh,
                                conf_thresh=conf_thresh,
                                enable_ood=True,
                            )
                            status = decision.get("status", "uncertain")
                            predictions = decision.get("predictions", predictions)
                        else:
                            top_conf = predictions[0]["confidence"] if predictions else 0.0
                            status = "certain" if top_conf >= conf_thresh else "uncertain"
                        st.session_state["last_result"] = {
                            "status": status,
                            "predictions": predictions,
                            "model_used": model_choice,
                        }
        with st.container(border=True):
            render_result(st.session_state.get("last_result"))

elif page == "Plant Care":
    st.markdown("<p class='page-intro'>Care notes come from the project’s existing botanical database. Missing fields are shown as unavailable.</p>", unsafe_allow_html=True)
    care_classes = [name for name in SUPPORTED_CLASSES if get_plant_profile(name)]
    if not care_classes:
        st.info("No care profiles are available for the active class mapping.")
    else:
        selected_plant = st.selectbox("Choose a supported plant", care_classes, key="care_species")
        profile = get_plant_profile(selected_plant)
        if profile:
            render_care_profile(profile)
        else:
            st.info("A care profile is not available for this species.")

elif page == "Supported Plants":
    st.markdown(
        f"<p class='page-intro'>These are the {len(SUPPORTED_CLASSES)} labels in the active checkpoint mapping. No external plant photographs are embedded.</p>",
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    for start in range(0, len(SUPPORTED_CLASSES), 3):
        columns = st.columns(3)
        for column, name in zip(columns, SUPPORTED_CLASSES[start:start + 3]):
            index = SUPPORTED_CLASSES.index(name)
            taxonomy = SPECIES_TAXONOMY.get(name, {})
            profile = get_plant_profile(name) or {}
            scientific_name = taxonomy.get("scientific_name") or profile.get("scientific_name") or "Scientific name unavailable"
            family = taxonomy.get("family") or profile.get("family") or "Family information unavailable"
            with column:
                with st.container(border=True):
                    st.markdown(
                        f"<div class='plant-placeholder'><span>{index + 1:02d}</span></div>"
                        f"<h3>{escape(name)}</h3><p class='scientific-name'>{escape(scientific_name)}</p>"
                        f"<p class='family-label'>{escape(family)}</p>",
                        unsafe_allow_html=True,
                    )

elif page == "Model Information":
    st.markdown("<p class='page-intro'>The default classifier is the trained EfficientNet-B0 checkpoint. Scores below are existing evaluation results, not a new evaluation.</p>", unsafe_allow_html=True)
    checkpoint_info = next(
        (entry for entry in InferenceEngine.list_available("./checkpoints") if entry["arch"] == "efficientnet_b0"),
        None,
    )
    if checkpoint_info and checkpoint_info["is_usable"]:
        st.success(f"EfficientNet-B0 checkpoint is available with {checkpoint_info['num_classes']} mapped classes.")
    else:
        st.warning("The EfficientNet-B0 checkpoint is not currently available.")

    report_path = Path(__file__).with_name("evaluation_report.json")
    evaluation = None
    if report_path.exists():
        try:
            with report_path.open("r", encoding="utf-8") as report_file:
                evaluation = json.load(report_file).get("efficientnet_b0")
        except (OSError, json.JSONDecodeError):
            evaluation = None
    if evaluation:
        metric_columns = st.columns(3)
        metrics = [
            ("Top-1 accuracy", evaluation.get("top1_acc")),
            ("Macro-F1", evaluation.get("macro_f1")),
            ("Held-out test images", evaluation.get("classification_report", {}).get("macro avg", {}).get("support")),
        ]
        for column, (label, value) in zip(metric_columns, metrics):
            with column:
                with st.container(border=True):
                    if label == "Held-out test images" and value is not None:
                        display_value = int(value)
                    elif value is not None:
                        display_value = f"{value}%"
                    else:
                        display_value = "Unavailable"
                    st.metric(label, display_value)
        st.caption("Recorded results: 81.88% top-1 accuracy and 81.49% macro-F1 on 977 held-out test images. These aggregate results do not guarantee accuracy for any individual species.")
    else:
        st.info("No verified EfficientNet-B0 evaluation record is available in evaluation_report.json.")
    st.markdown("#### Inference pipeline")
    st.markdown("Image quality screening → selected CNN checkpoint → existing confidence/OOD handling → ranked class candidates.")
    st.caption("The inference pipeline and checkpoint settings are unchanged by this UI update.")

elif page == "About Project":
    st.markdown("<p class='page-intro'>A local college-project demonstration combining image classification with a curated indoor-plant care reference.</p>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### Project objective")
            st.write("Explore how transfer learning can help identify a limited set of indoor plants and present useful care references in one interface.")
        with st.container(border=True):
            st.markdown("### Workflow")
            st.write("Organized plant images are split for training and evaluation. A trained checkpoint supplies ranked class scores; image screening and uncertainty handling remain part of inference.")
    with right:
        with st.container(border=True):
            st.markdown("### Technology")
            st.write("Streamlit · Python · PyTorch · Torchvision · EfficientNet-B0 · Pillow")
        with st.container(border=True):
            st.markdown("### Limitations")
            st.write("The active model maps 19 classes. Image labels and dataset coverage may be imperfect. The application does not diagnose disease, and confidence is not a guarantee. Use pet-safety information as a reference and verify important decisions with qualified sources.")
    st.caption("This demonstration runs image inference locally and does not send uploaded photos to an external service.")
