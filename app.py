"""Streamlit plant-identification interface for locally stored CNN checkpoints."""

import io
import json
from html import escape
from pathlib import Path
import math
import streamlit as st
from PIL import Image, UnidentifiedImageError

from model import InferenceEngine, load_inference_engine, resolve_species_metadata
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
@st.cache_resource(show_spinner="Loading model checkpoint...")
def get_model_engine(checkpoint_dir, arch, signature):
    engine, err = load_inference_engine(checkpoint_dir, arch)
    return engine, err


PAGES = (
    "Home",
    "Identify Plant",
    "Plant Care",
    "Supported Plants",
    "Model Performance",
    "Model Information",
    "About Project",
)
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIRS = (
    PROJECT_ROOT / "checkpoints_47",
    PROJECT_ROOT / "checkpoints",
)


def checkpoint_signature(checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir)
    candidates = [
        checkpoint_path / "class_mapping.json",
        checkpoint_path / "efficientnet_b0_best.pt",
        checkpoint_path / "mobilenet_v3_small_best.pt",
        checkpoint_path / "ood_thresholds.json",
        checkpoint_path / "preprocessing_config.json",
        checkpoint_path / "evaluation_report.json",
        checkpoint_path / "training_metadata.json",
    ]
    return tuple(
        (
            str(path),
            path.stat().st_mtime_ns if path.exists() else None,
            path.stat().st_size if path.exists() else None,
        )
        for path in candidates
    )


@st.cache_data(max_entries=8, show_spinner=False)
def list_checkpoint_diagnostics(checkpoint_dir, signature):
    return InferenceEngine.list_available(checkpoint_dir)


@st.cache_data(max_entries=8, show_spinner=False)
def read_json_file(file_path, signature):
    with open(file_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def load_checkpoint_json(checkpoint_dir, filename):
    file_path = Path(checkpoint_dir) / filename
    if not file_path.is_file():
        return None
    stat = file_path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    try:
        return read_json_file(str(file_path), signature)
    except (OSError, json.JSONDecodeError) as error:
        st.warning(f"Saved artifact '{file_path.name}' could not be read: {error}")
        return None


def load_evaluation(checkpoint_dir, arch):
    report = load_checkpoint_json(checkpoint_dir, "evaluation_report.json")
    if (
        report is None
        and Path(checkpoint_dir).resolve() == (PROJECT_ROOT / "checkpoints").resolve()
    ):
        report = load_checkpoint_json(str(PROJECT_ROOT), "evaluation_report.json")
    if not isinstance(report, dict):
        return None
    evaluation = report.get(arch)
    if not isinstance(evaluation, dict) and report.get("arch") == arch:
        evaluation = report
    if not isinstance(evaluation, dict):
        return None
    report_classes = set(evaluation.get("dataset_report", {}).get("class_names", []))
    if not report_classes:
        report_classes = {
            label
            for label in evaluation.get("classification_report", {})
            if label not in {"accuracy", "macro avg", "weighted avg"}
        }
    if report_classes and report_classes != set(load_class_names(checkpoint_dir)):
        return None
    return evaluation


def display_architecture(arch):
    return {
        "efficientnet_b0": "EfficientNet-B0",
        "mobilenet_v3_small": "MobileNetV3-Small",
    }.get(arch, arch.replace("_", " ").title())


def discover_model_options():
    options = {}
    for checkpoint_dir in CHECKPOINT_DIRS:
        signature = checkpoint_signature(checkpoint_dir)
        for entry in list_checkpoint_diagnostics(str(checkpoint_dir), signature):
            if not entry["is_usable"]:
                continue
            arch = entry["arch"]
            label = f"{display_architecture(arch)} · {entry['num_classes']} classes"
            if label in options:
                label = f"{label} · {checkpoint_dir.name}"
            options[label] = {
                "arch": arch,
                "checkpoint_dir": str(checkpoint_dir),
                "num_classes": entry["num_classes"],
                "checkpoint_signature": signature,
            }
    return options


def load_class_names(checkpoint_dir):
    mapping_path = Path(checkpoint_dir) / "class_mapping.json"
    with mapping_path.open("r", encoding="utf-8") as mapping_file:
        raw_mapping = json.load(mapping_file).get("id2label", {})
    labels = {int(index): label for index, label in raw_mapping.items()}
    if not labels or sorted(labels) != list(range(len(labels))):
        raise ValueError(f"Invalid class mapping in '{mapping_path}'.")
    return [labels[index] for index in range(len(labels))]


MODEL_OPTIONS = discover_model_options()
MODEL_OPTION_LABELS = list(MODEL_OPTIONS)
DEFAULT_MODEL_INDEX = next(
    (
        index for index, label in enumerate(MODEL_OPTION_LABELS)
        if MODEL_OPTIONS[label]["checkpoint_dir"] == str(CHECKPOINT_DIRS[0])
        and MODEL_OPTIONS[label]["arch"] == "efficientnet_b0"
    ),
    next(
        (
            index for index, label in enumerate(MODEL_OPTION_LABELS)
            if MODEL_OPTIONS[label]["checkpoint_dir"] == str(CHECKPOINT_DIRS[1])
            and MODEL_OPTIONS[label]["arch"] == "efficientnet_b0"
        ),
        0,
    ),
)


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
    elif result.get("status") == "uncertain":
        st.info("No care profile is available for this unconfirmed candidate.")
    else:
        st.info("No care profile is available for this plant class.")


with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'><span class='brand-mark'>F</span>"
        "<span><strong>FloraSense</strong><small>Plant intelligence</small></span></div>",
        unsafe_allow_html=True,
    )
    page = st.radio("Navigate", PAGES, key="navigation", label_visibility="collapsed")
    st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)
    st.markdown("#### Inference settings")
    if MODEL_OPTION_LABELS:
        model_choice = st.selectbox(
            "Model checkpoint",
            MODEL_OPTION_LABELS,
            index=DEFAULT_MODEL_INDEX,
            key="selected_checkpoint",
        )
        selected_model = MODEL_OPTIONS[model_choice]
        checkpoint_dir = selected_model["checkpoint_dir"]
        selected_arch = selected_model["arch"]
        selected_signature = selected_model["checkpoint_signature"]
        selected_classes = load_class_names(checkpoint_dir)
        default_thresholds = load_thresholds(
            str(Path(checkpoint_dir) / "ood_thresholds.json")
        )
        active_model_key = f"{checkpoint_dir}|{selected_arch}|{selected_signature}"
        if st.session_state.get("active_model_key") != active_model_key:
            st.session_state["last_result"] = None
            st.session_state["active_model_key"] = active_model_key
    else:
        model_choice = None
        selected_model = None
        checkpoint_dir = str(CHECKPOINT_DIRS[0])
        selected_arch = "efficientnet_b0"
        selected_signature = checkpoint_signature(checkpoint_dir)
        selected_classes = []
        default_thresholds = load_thresholds(
            str(CHECKPOINT_DIRS[0] / "ood_thresholds.json")
        )
        st.warning("No usable model checkpoints were found.")
    SUPPORTED_CLASSES = selected_classes
    model_widget_suffix = (
        f"{Path(checkpoint_dir).name}_{selected_arch}_{selected_model['num_classes']}"
        if selected_model else "unavailable"
    )
    with st.expander("Advanced settings", expanded=False):
        enable_ood = st.checkbox(
            "Enable OOD filtering",
            value=True,
            key=f"enable_ood_{model_widget_suffix}",
        )
        configured_energy = float(default_thresholds["energy_thresh"])
        energy_thresh = st.slider(
            "Energy threshold",
            min_value=float(min(-40.0, math.floor(configured_energy - 10.0))),
            max_value=float(max(-5.0, math.ceil(configured_energy + 10.0))),
            value=configured_energy,
            step=0.1,
            key=f"energy_threshold_{model_widget_suffix}",
            help="Higher values demand stronger evidence before accepting a prediction as certain.",
        )
        conf_thresh = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(default_thresholds["conf_thresh"]),
            step=0.01,
            key=f"confidence_threshold_{model_widget_suffix}",
            help="Minimum top-1 probability required to confirm a prediction.",
        )
        margin_thresh = st.slider(
            "Top-two margin threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(default_thresholds["margin_thresh"]),
            step=0.01,
            key=f"margin_threshold_{model_widget_suffix}",
            help="Minimum probability gap required between the top two candidates.",
        )
        st.markdown("#### Checkpoint diagnostics")
        for checkpoint_dir_path in CHECKPOINT_DIRS:
            signature = checkpoint_signature(checkpoint_dir_path)
            for checkpoint in list_checkpoint_diagnostics(str(checkpoint_dir_path), signature):
                st.markdown(
                    f"**{display_architecture(checkpoint['arch'])} · {checkpoint_dir_path.name}**"
                )
                if checkpoint["is_usable"]:
                    st.caption(f"Ready · {checkpoint['num_classes']} mapped classes")
                else:
                    st.caption(checkpoint.get("error") or "Checkpoint unavailable")
    temperature = float(default_thresholds["temperature"])

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
    evaluation = load_evaluation(checkpoint_dir, selected_arch)
    evaluation_report = evaluation or {}
    test_images = evaluation_report.get("dataset_report", {}).get("total_images")
    if test_images is None:
        test_images = (
            evaluation_report.get("classification_report", {})
            .get("weighted avg", {})
            .get("support")
        )
    home_metrics = st.columns(4)
    verified_stats = (
        ("Mapped classes", len(SUPPORTED_CLASSES)),
        (
            "Held-out top-1",
            f"{evaluation_report['top1_acc']:.2f}%"
            if isinstance(evaluation_report.get("top1_acc"), (int, float))
            else "Not available",
        ),
        (
            "Held-out test images",
            int(test_images) if isinstance(test_images, (int, float)) else "Not available",
        ),
        ("Care profiles", sum(bool(get_plant_profile(name)) for name in SUPPORTED_CLASSES)),
    )
    for column, (label, value) in zip(home_metrics, verified_stats):
        with column:
            with st.container(border=True):
                st.metric(label, value)
    if evaluation:
        st.caption("Held-out top-1 is the saved offline evaluation score; it is not an individual prediction confidence.")
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
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    st.error(f"This image could not be opened: {error}")
            identify_col, reset_col = st.columns(2)
            with identify_col:
                run_classify = st.button(
                    "Identify",
                    type="primary",
                    use_container_width=True,
                    disabled=image is None or selected_model is None,
                )
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
                    try:
                        engine, error = get_model_engine(
                            checkpoint_dir, selected_arch, selected_signature
                        )
                        if engine is None:
                            st.error(f"Model checkpoint unavailable: {error}")
                            st.session_state["last_result"] = None
                        else:
                            logits, predictions = engine.predict(image, top_k=5)
                            decision = calibrate_and_filter(
                                logits=logits,
                                predictions=predictions,
                                temperature=temperature,
                                energy_thresh=energy_thresh,
                                conf_thresh=conf_thresh,
                                margin_thresh=margin_thresh,
                                enable_ood=enable_ood,
                            )
                            status = decision.get("status", "uncertain")
                            predictions = decision.get("predictions", predictions)
                            st.session_state["last_result"] = {
                                "status": status,
                                "predictions": predictions,
                                "model_used": model_choice,
                            }
                    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
                        st.error(f"Prediction failed: {error}")
                        st.session_state["last_result"] = None
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
    plant_query = st.text_input("Search supported plants", placeholder="Search by common or scientific name")
    filtered_classes = [
        name for name in SUPPORTED_CLASSES
        if plant_query.casefold() in name.casefold()
        or plant_query.casefold() in resolve_species_metadata(name)[0].casefold()
        or plant_query.casefold() in resolve_species_metadata(name)[1].casefold()
    ]
    st.caption(f"Showing {len(filtered_classes)} of {len(SUPPORTED_CLASSES)} mapped classes.")
    if not filtered_classes:
        st.info("No supported plants match this search.")
    for start in range(0, len(filtered_classes), 3):
        columns = st.columns(3)
        for column, name in zip(columns, filtered_classes[start:start + 3]):
            scientific_name, common_name = resolve_species_metadata(name)
            profile = get_plant_profile(name) or get_plant_profile(common_name) or {}
            family = profile.get("family") or "Family information unavailable"
            index = SUPPORTED_CLASSES.index(name)
            with column:
                with st.container(border=True):
                    st.markdown(
                        f"<div class='plant-placeholder'><span>{index + 1:02d}</span></div>"
                        f"<h3>{escape(common_name)}</h3><p class='scientific-name'>{escape(scientific_name)}</p>"
                        f"<p class='family-label'>{escape(family)}</p>",
                        unsafe_allow_html=True,
                    )

elif page == "Model Performance":
    st.markdown(
        "<p class='page-intro'>Saved offline evaluation for the selected checkpoint. "
        "These measurements are separate from confidence scores shown for an individual photo.</p>",
        unsafe_allow_html=True,
    )
    evaluation = load_evaluation(checkpoint_dir, selected_arch)
    if not evaluation:
        st.info("No matching saved test-set evaluation report is available for this checkpoint.")
    else:
        class_report = evaluation.get("classification_report", {})
        dataset_report = evaluation.get("dataset_report", {})
        test_count = dataset_report.get("total_images")
        if test_count is None:
            test_count = class_report.get("weighted avg", {}).get("support")
        metric_values = [
            ("Top-1 accuracy", evaluation.get("top1_acc")),
            ("Top-3 accuracy", evaluation.get("top3_acc")),
            ("Macro-F1", evaluation.get("macro_f1")),
            (
                "Weighted-F1",
                evaluation.get("weighted_f1")
                if isinstance(evaluation.get("weighted_f1"), (int, float))
                else (
                    class_report.get("weighted avg", {}).get("f1-score", 0) * 100
                    if isinstance(class_report.get("weighted avg", {}).get("f1-score"), (int, float))
                    else None
                ),
            ),
            ("Test images", test_count),
        ]
        metric_columns = st.columns(len(metric_values))
        for column, (label, value) in zip(metric_columns, metric_values):
            with column:
                if label == "Test images":
                    display_value = int(value) if isinstance(value, (int, float)) else "Unavailable"
                else:
                    display_value = f"{value:.2f}%" if isinstance(value, (int, float)) else "Unavailable"
                st.metric(label, display_value)
        st.caption(
            "Saved held-out test metrics are read from the evaluation report paired with this checkpoint. "
            "They are not recalculated on app launch and must not be interpreted as confidence for a single prediction."
        )

        per_class_rows = []
        for plant_name in selected_classes:
            class_metrics = class_report.get(plant_name)
            if not isinstance(class_metrics, dict):
                continue
            per_class_rows.append({
                "Plant class": plant_name,
                "Precision": round(class_metrics.get("precision", 0), 4),
                "Recall": round(class_metrics.get("recall", 0), 4),
                "F1": round(class_metrics.get("f1-score", 0), 4),
                "Support": int(class_metrics.get("support", 0)),
            })
        st.markdown("### Per-class performance")
        if per_class_rows:
            st.dataframe(per_class_rows, use_container_width=True, hide_index=True)
        else:
            st.info("Per-class precision and recall are not present in this report.")

        confusion = evaluation.get("confusion_matrix", {})
        matrix = confusion.get("matrix") if isinstance(confusion, dict) else None
        confusion_labels = confusion.get("labels", []) if isinstance(confusion, dict) else []
        valid_matrix = (
            isinstance(matrix, list)
            and confusion_labels == selected_classes
            and len(matrix) == len(selected_classes)
            and all(isinstance(row, list) and len(row) == len(selected_classes) for row in matrix)
        )
        st.markdown("### Confusion matrix")
        confusion_image = Path(checkpoint_dir) / f"{selected_arch}_confusion_matrix.png"
        if confusion_image.is_file():
            st.image(str(confusion_image), caption="Saved confusion matrix from the selected checkpoint's held-out test evaluation.")
        elif valid_matrix:
            with st.expander("View confusion matrix values"):
                st.dataframe(
                    [
                        {"Actual class": label, **{
                            f"Predicted: {predicted}": count
                            for predicted, count in zip(selected_classes, row)
                        }}
                        for label, row in zip(selected_classes, matrix)
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("A class-aligned confusion matrix is not available for this checkpoint.")

elif page == "Model Information":
    st.markdown(
        "<p class='page-intro'>Checkpoint, preprocessing, and training details for the active model.</p>",
        unsafe_allow_html=True,
    )
    diagnostics = list_checkpoint_diagnostics(checkpoint_dir, selected_signature)
    checkpoint_info = next(
        (entry for entry in diagnostics if entry["arch"] == selected_arch),
        None,
    )
    if checkpoint_info and checkpoint_info["is_usable"]:
        st.success(
            f"{display_architecture(selected_arch)} is ready with "
            f"{checkpoint_info['num_classes']} mapped classes."
        )
        st.write(f"Checkpoint directory: `{Path(checkpoint_dir).name}`")
        st.write(f"Output labels: **{len(selected_classes)}** (from the matching `class_mapping.json`).")
    else:
        st.warning(
            checkpoint_info.get("error", "The selected model checkpoint is not currently available.")
            if checkpoint_info else "No matching model entry was found."
        )

    preprocessing = load_checkpoint_json(checkpoint_dir, "preprocessing_config.json")
    metadata = load_checkpoint_json(checkpoint_dir, "training_metadata.json")
    st.markdown("### Inference workflow")
    st.write(
        "RGB upload → image-quality screen → checkpoint-matched resize/crop/normalization "
        "→ CNN scores → existing energy, confidence, and top-two-margin filtering."
    )
    if preprocessing:
        st.markdown("### Preprocessing configuration")
        st.json(preprocessing)
    else:
        st.info("No checkpoint-specific preprocessing configuration was found.")
    if metadata:
        st.markdown("### Training metadata")
        metadata_summary = {
            key: metadata[key]
            for key in (
                "arch", "num_classes", "train_samples", "validation_samples",
                "test_samples", "warmup_epochs", "finetune_epochs",
                "best_acc", "is_demo_run", "test_used_for_model_selection",
                "test_used_for_ood_calibration",
            )
            if key in metadata
        }
        st.json(metadata_summary)
    else:
        st.info("No training metadata was found for this checkpoint.")
    st.markdown("### Limitations")
    st.write(
        "This classifier only ranks labels in the selected mapping. It may be wrong for "
        "unsupported species, unusual views, or images unlike the training data. Confidence "
        "is not a guarantee, and care notes are references rather than diagnosis."
    )
    st.caption(
        "Checkpoint diagnostics, mappings, preprocessing, and OOD thresholds remain paired "
        "with their respective model folders. Unavailable checkpoints are not offered for selection."
    )

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
            st.write(f"The selected checkpoint maps {len(SUPPORTED_CLASSES)} classes. Image labels and dataset coverage may be imperfect. The application does not diagnose disease, and confidence is not a guarantee. Use pet-safety information as a reference and verify important decisions with qualified sources.")
    st.caption("This demonstration runs image inference locally and does not send uploaded photos to an external service.")
