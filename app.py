# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from PIL import Image

# Import từ main.py và modules
from main import run_experiment, experiment_models, preprocess_text, vectorize_text
from modules.data_processing import load_data, split_data, encode_labels
from modules.models import evaluate_model, print_evaluation_results
from modules.config import RESULTS_DIR
import streamlit.components.v1 as components

# Tiêu đề ứng dụng và giao diện
st.set_page_config(page_title="HOMO-LAT Sentiment Analysis Tool", layout="wide")
st.title("HOMO-LAT Sentiment Analysis Experiment Runner")
st.markdown("""
    Công cụ khảo sát các phương pháp và mô hình phân loại cảm xúc cho dữ liệu HOMO-LAT.
    Cho phép thử nghiệm với nhiều phương pháp tiền xử lý, kỹ thuật vector hóa và fine-tuning tham số.
""")

# Tạo tabs cho giao diện
tab1, tab2, tab3, tab4 = st.tabs(["Tiền xử lý & Mô hình", "Fine-tune & Tối ưu tham số", "So sánh mô hình", "Dự đoán"])

# Tab 1: Tiền xử lý & Mô hình cơ bản
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Tiền xử lý dữ liệu")
        
        # Chọn phương pháp tiền xử lý
        preprocessing_pipeline = st.selectbox(
            "Phương pháp tiền xử lý:",
            ["basic", "normalize_accents", "remove_stopwords", "stemming", "lemmatization", "full"],
            format_func=lambda x: {
                "basic": "Làm sạch cơ bản",
                "normalize_accents": "Chuẩn hóa dấu",
                "remove_stopwords": "Loại bỏ stopwords",
                "stemming": "Stemming",
                "lemmatization": "Lemmatization",
                "full": "Áp dụng tất cả"
            }.get(x, x)
        )
        
        # Tham số cho N-gram
        st.subheader("Tham số N-gram")
        ngram_min = st.number_input("N-gram tối thiểu", min_value=1, value=1)
        ngram_max = st.number_input("N-gram tối đa", min_value=1, value=2)
        
    with col2:
        st.subheader("Vector hóa & Mô hình")
        
        # Chọn phương pháp vector hóa
        vectorization_method = st.selectbox(
            "Phương pháp vector hóa:",
            ["bow", "tfidf", "word2vec", "fasttext"],
            format_func=lambda x: {
                "bow": "Bag of Words",
                "tfidf": "TF-IDF",
                "word2vec": "Word2Vec",
                "fasttext": "FastText"
            }.get(x, x)
        )
        
        # Tham số bổ sung cho phương pháp nhúng từ
        if vectorization_method in ["word2vec", "fasttext"]:
            embedding_dim = st.number_input(
                "Kích thước embedding vector", 
                min_value=50, max_value=300, value=100, step=50
            )
        else:
            embedding_dim = 100
            
        # Lựa chọn mô hình
        model_choice = st.selectbox(
            "Mô hình phân loại:",
            ["naive_bayes", "logistic_regression", "svm", "linear_svc", "random_forest",
             "gradient_boosting", "knn", "decision_tree"],
            format_func=lambda x: {
                "naive_bayes": "Naive Bayes",
                "logistic_regression": "Logistic Regression",
                "svm": "Support Vector Machine",
                "linear_svc": "Linear SVC",
                "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting",
                "knn": "K-Nearest Neighbors",
                "decision_tree": "Decision Tree"
            }.get(x, x)
        )
        
    # Các tùy chọn bổ sung
    st.subheader("Tùy chọn bổ sung")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        use_dev_set = st.checkbox("Sử dụng tập dev cho đánh giá", value=False)
    
    with col2:
        tune = st.checkbox("Thực hiện fine-tune tham số cơ bản", value=False)
    
    with col3:
        balance = st.checkbox("Xử lý bất cân bằng dữ liệu", value=False)
    
    # Nút chạy thí nghiệm cơ bản
    if st.button("Chạy thí nghiệm cơ bản"):
        with st.spinner("Đang chạy thí nghiệm..."):
            try:
                experiment_results = run_experiment(
                    preprocessing_pipeline=preprocessing_pipeline,
                    vectorization_method=vectorization_method,
                    model_name=model_choice,
                    use_dev_set=use_dev_set,
                    use_hyperparameter_tuning=tune,
                    handle_class_imbalance=balance,
                    ngram_range=(ngram_min, ngram_max),
                    embedding_dim=embedding_dim,
                    save_results=True  # Lưu kết quả
                )
                
                st.success("Thí nghiệm hoàn thành!")
                
                # Hiển thị kết quả
                results = experiment_results["results"]
                
                # Hiển thị metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{results['accuracy']:.3f}")
                col2.metric("F1 Macro", f"{results['f1_macro']:.3f}")
                col3.metric("F1 Weighted", f"{results['f1_weighted']:.3f}")
                
                # Hiển thị confusion matrix
                st.subheader("Ma trận nhầm lẫn")
                fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', ax=ax)
                plt.ylabel('Thực tế')
                plt.xlabel('Dự đoán')
                st.pyplot(fig)
                
                # Hiển thị báo cáo phân loại
                st.subheader("Báo cáo phân loại chi tiết")
                st.json(results.get("classification_report", {}))
                
                # Lưu kết quả vào session state
                st.session_state.last_experiment = experiment_results
                st.session_state.last_experiment_name = experiment_results["experiment_name"]
                st.session_state.model = experiment_results["model"]
                st.session_state.vectorizer = experiment_results["vectorizer"]
                
            except Exception as e:
                st.error(f"Lỗi khi chạy thí nghiệm: {str(e)}")

# Tab 2: Fine-tune và Tối ưu tham số
with tab2:
    st.subheader("Tinh chỉnh tham số toàn diện")
    
    # Giải thích
    st.markdown("""
        Tab này cho phép bạn thực hiện tìm kiếm lưới tham số (GridSearchCV) toàn diện cho tất cả các mô hình
        để tìm ra mô hình và bộ tham số tối ưu nhất. Quá trình này có thể mất nhiều thời gian, nhưng sẽ cho
        kết quả chi tiết về hiệu suất của từng mô hình với nhiều cấu hình tham số khác nhau.
    """)
    
    # Thiết lập cho thử nghiệm nhiều tham số
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Chọn phương pháp tiền xử lý
        preprocessing_pipeline_ft = st.selectbox(
            "Phương pháp tiền xử lý cho fine-tuning:",
            ["basic", "normalize_accents", "remove_stopwords", "stemming", "lemmatization", "full"],
            format_func=lambda x: {
                "basic": "Làm sạch cơ bản",
                "normalize_accents": "Chuẩn hóa dấu",
                "remove_stopwords": "Loại bỏ stopwords",
                "stemming": "Stemming",
                "lemmatization": "Lemmatization",
                "full": "Áp dụng tất cả"
            }.get(x, x),
            key="preprocess_ft"
        )
        
        # Chọn phương pháp vector hóa
        vectorization_method_ft = st.selectbox(
            "Phương pháp vector hóa cho fine-tuning:",
            ["bow", "tfidf", "word2vec", "fasttext"],
            format_func=lambda x: {
                "bow": "Bag of Words",
                "tfidf": "TF-IDF",
                "word2vec": "Word2Vec",
                "fasttext": "FastText"
            }.get(x, x),
            key="vectorize_ft"
        )
        
    with col2:
        # Tùy chọn N-gram 
        ngram_min_ft = st.slider("N-gram tối thiểu cho fine-tuning", 1, 3, 1)
        ngram_max_ft = st.slider("N-gram tối đa cho fine-tuning", 1, 5, 2)
        
        # Embedding dimension nếu cần
        if vectorization_method_ft in ["word2vec", "fasttext"]:
            embedding_dim_ft = st.slider(
                "Kích thước embedding vector cho fine-tuning", 
                50, 300, 100, 50
            )
        else:
            embedding_dim_ft = 100
    
    # Chọn mô hình cụ thể để fine-tune
    fine_tune_all = st.checkbox("Fine-tune tất cả các mô hình", value=True)
    
    if not fine_tune_all:
        models_to_finetune = st.multiselect(
            "Chọn các mô hình để fine-tune:",
            ["naive_bayes", "logistic_regression", "svm", "linear_svc", 
             "random_forest", "gradient_boosting", "knn", "decision_tree"],
            default=["logistic_regression", "svm"],
            format_func=lambda x: {
                "naive_bayes": "Naive Bayes",
                "logistic_regression": "Logistic Regression",
                "svm": "Support Vector Machine",
                "linear_svc": "Linear SVC",
                "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting",
                "knn": "K-Nearest Neighbors",
                "decision_tree": "Decision Tree"
            }.get(x, x)
        )
    else:
        models_to_finetune = ["naive_bayes", "logistic_regression", "svm", "linear_svc", 
                            "random_forest", "gradient_boosting", "knn", "decision_tree"]
    
    # Nút để chạy fine-tuning
    if st.button("Chạy Fine-tuning toàn diện"):
        with st.spinner("Đang thực hiện fine-tuning toàn diện với nhiều tham số... Quá trình này có thể mất vài phút đến vài giờ tùy thuộc vào số lượng mô hình và tham số cần thử nghiệm."):
            try:
                # Load dữ liệu
                train_df, dev_df = load_data()
                
                # Phân chia dữ liệu
                if use_dev_set:
                    X_train, y_train = train_df['post content'], train_df['label']
                    X_val, y_val = dev_df['post content'], dev_df['label']
                else:
                    train_df, val_df = split_data(train_df)
                    X_train, y_train = train_df['post content'], train_df['label']
                    X_val, y_val = val_df['post content'], val_df['label']
                
                # Encode labels
                y_train_encoded, y_val_encoded, label_encoder = encode_labels(y_train, y_val)
                class_names = label_encoder.classes_
                
                # Tiền xử lý văn bản
                X_train_preprocessed = preprocess_text(X_train, preprocessing_pipeline_ft)
                X_val_preprocessed = preprocess_text(X_val, preprocessing_pipeline_ft)
                
                # Tạo embedding model nếu cần
                embedding_model = None
                if vectorization_method_ft in ['word2vec', 'fasttext']:
                    from modules.feature_engineering import create_word2vec_model, create_fasttext_model
                    tokenized_sentences = [text.split() for text in X_train_preprocessed if isinstance(text, str)]
                    
                    if vectorization_method_ft == 'word2vec':
                        embedding_model = create_word2vec_model(
                            sentences=tokenized_sentences, 
                            vector_size=embedding_dim_ft
                        )
                    elif vectorization_method_ft == 'fasttext':
                        embedding_model = create_fasttext_model(
                            sentences=tokenized_sentences, 
                            vector_size=embedding_dim_ft
                        )
                
                # Vector hóa văn bản
                X_train_vectorized, vectorizer = vectorize_text(
                    X_train_preprocessed, vectorization_method_ft,
                    ngram_range=(ngram_min_ft, ngram_max_ft), 
                    embedding_model=embedding_model
                )
                
                # Transform dữ liệu validation
                if vectorization_method_ft in ['word2vec', 'fasttext']:
                    X_val_vectorized = vectorizer.transform(X_val_preprocessed)
                else:
                    X_val_vectorized = vectorizer.transform(X_val_preprocessed)
                
                # Thực hiện fine-tuning
                experiment_results = experiment_models(
                    X_train=X_train_vectorized, 
                    y_train=y_train_encoded, 
                    X_val=X_val_vectorized, 
                    y_val=y_val_encoded, 
                    class_names=class_names
                )
                
                st.success("Fine-tuning hoàn thành!")
                
                # Lưu kết quả vào session state
                st.session_state.experiment_results = experiment_results
                
                # Hiển thị kết quả so sánh
                comparison_df = experiment_results['comparison_df']
                
                # Hiển thị dữ liệu so sánh dưới dạng bảng
                st.subheader("Bảng so sánh các mô hình")
                st.dataframe(comparison_df)
                
                # Hiển thị biểu đồ so sánh
                st.subheader("So sánh hiệu suất các mô hình")
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(x='model', y='val_f1_macro', data=comparison_df, ax=ax)
                plt.title('So sánh F1-Macro trên tập validation')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Hiển thị thông tin về mô hình tốt nhất
                st.subheader(f"Mô hình tốt nhất: {experiment_results['best_model_name'].upper()}")
                st.write(f"Tham số tốt nhất:")
                for param, value in experiment_results['best_params'].items():
                    st.write(f"- **{param}:** {value}")
                
                # Lưu thông tin về đường dẫn đến kết quả
                st.info(f"Kết quả chi tiết được lưu tại: {experiment_results['experiment_dir']}")
                
            except Exception as e:
                st.error(f"Lỗi khi thực hiện fine-tuning: {str(e)}")

# Tab 3: So sánh mô hình
with tab3:
    st.subheader("So sánh các mô hình")
    
    # Tìm tất cả các thư mục kết quả
    all_results_dirs = []
    if os.path.exists(RESULTS_DIR):
        for item in os.listdir(RESULTS_DIR):
            item_path = os.path.join(RESULTS_DIR, item)
            if os.path.isdir(item_path):
                all_results_dirs.append(item)
    
    if all_results_dirs:
        st.markdown("""
            Tab này cho phép bạn so sánh các mô hình đã huấn luyện trước đó. 
            Chọn các thí nghiệm bạn muốn so sánh từ danh sách dưới đây.
        """)
        
        selected_experiments = st.multiselect(
            "Chọn thí nghiệm để so sánh:",
            all_results_dirs
        )
        
        if selected_experiments and st.button("So sánh các thí nghiệm"):
            with st.spinner("Đang tải và so sánh các thí nghiệm..."):
                try:
                    # Thu thập kết quả từ các thí nghiệm đã chọn
                    comparison_data = []
                    
                    for exp_name in selected_experiments:
                        exp_dir = os.path.join(RESULTS_DIR, exp_name)
                        
                        # Tìm file results.json
                        results_file = None
                        for file in os.listdir(exp_dir):
                            if file.endswith("_results.json"):
                                results_file = os.path.join(exp_dir, file)
                                break
                        
                        if results_file:
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                                
                                # Trích xuất thông tin cấu hình
                                config = results.get('experiment_config', {})
                                
                                comparison_data.append({
                                    'experiment': exp_name,
                                    'model': config.get('model_name', 'unknown'),
                                    'preprocessing': config.get('preprocessing_pipeline', 'unknown'),
                                    'vectorization': config.get('vectorization_method', 'unknown'),
                                    'accuracy': results.get('accuracy', 0),
                                    'f1_macro': results.get('f1_macro', 0),
                                    'f1_weighted': results.get('f1_weighted', 0),
                                    'execution_time': config.get('execution_time', 0)
                                })
                    
                    if comparison_data:
                        # Tạo DataFrame từ dữ liệu
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Hiển thị bảng so sánh
                        st.subheader("So sánh hiệu suất các thí nghiệm")
                        st.dataframe(comparison_df)
                        
                        # Hiển thị biểu đồ so sánh
                        st.subheader("So sánh các metrics")
                        
                        # Biểu đồ cột cho F1-macro
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(data=comparison_df, x='experiment', y='f1_macro', ax=ax)
                        plt.title('So sánh F1-Macro')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Biểu đồ so sánh 3 metrics
                        fig, ax = plt.subplots(figsize=(12, 6))
                        comp_melted = pd.melt(comparison_df, 
                                            id_vars=['experiment'], 
                                            value_vars=['accuracy', 'f1_macro', 'f1_weighted'],
                                            var_name='metric', value_name='score')
                        sns.barplot(data=comp_melted, x='experiment', y='score', hue='metric', ax=ax)
                        plt.title('So sánh các Metrics')
                        plt.xticks(rotation=45)
                        plt.legend(title='Metric')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    else:
                        st.warning("Không thể tìm thấy kết quả cho thí nghiệm đã chọn.")
                
                except Exception as e:
                    st.error(f"Lỗi khi so sánh các thí nghiệm: {str(e)}")
    else:
        st.info("Chưa có thí nghiệm nào được chạy. Hãy chạy một số thí nghiệm ở tab 'Tiền xử lý & Mô hình' hoặc 'Fine-tune & Tối ưu tham số' trước.")
        
    # Hiển thị chi tiết tham số mô hình
    if 'experiment_results' in st.session_state:
        st.subheader("Chi tiết tham số từ lần fine-tune gần nhất")
        exp_results = st.session_state.experiment_results
        
        # Danh sách các mô hình
        models = list(exp_results['all_models'].keys())
        
        # Chọn mô hình để xem chi tiết tham số
        selected_model = st.selectbox("Chọn mô hình để xem chi tiết tham số:", models)
        
        if selected_model:
            model_info = exp_results['all_models'][selected_model]
            st.write(f"**Tham số tốt nhất cho {selected_model}:**")
            for param, value in model_info['params'].items():
                st.write(f"- **{param}:** {value}")
                
            # Hiển thị metrics
            st.write("\n**Kết quả đánh giá:**")
            val_score = model_info['val_score']
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{val_score['accuracy']:.3f}")
            col2.metric("F1 Macro", f"{val_score['f1_macro']:.3f}")
            col3.metric("F1 Weighted", f"{val_score['f1_weighted']:.3f}")
            
            # Hiển thị confusion matrix
            st.write("\n**Confusion Matrix:**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(val_score['confusion_matrix'], annot=True, fmt='d', ax=ax)
            plt.title(f'Confusion Matrix cho {selected_model}')
            plt.ylabel('Thực tế')
            plt.xlabel('Dự đoán')
            st.pyplot(fig)

# Tab 4: Dự đoán
with tab4:
    st.subheader("Dự đoán văn bản mới")
    
    # Kiểm tra xem có model trong session state không
    if 'model' in st.session_state and 'vectorizer' in st.session_state:
        # Text area cho người dùng nhập văn bản
        user_input = st.text_area("Nhập văn bản cần phân loại:", height=150)
        
        # Nút dự đoán
        if user_input and st.button("Dự đoán"):
            try:
                # Tiền xử lý văn bản
                if 'last_experiment' in st.session_state:
                    last_exp = st.session_state.last_experiment
                    preprocessing_method = last_exp.get('experiment_config', {}).get('preprocessing_pipeline', 'basic')
                else:
                    preprocessing_method = 'basic'
                
                # Tiền xử lý
                processed_text = preprocess_text([user_input], preprocessing_method)[0]
                
                # Vector hóa
                vectorizer = st.session_state.vectorizer
                
                if hasattr(vectorizer, 'transform'):
                    X = vectorizer.transform([processed_text])
                else:  # Cho embedding vectorizers
                    X = vectorizer.transform([processed_text])
                
                # Dự đoán
                model = st.session_state.model
                prediction = model.predict(X)[0]
                
                # Lấy tên lớp nếu có encoder
                if hasattr(model, 'classes_'):
                    class_name = model.classes_[prediction]
                else:
                    class_name = str(prediction)
                
                # Hiển thị kết quả
                st.success(f"Kết quả dự đoán: **{class_name}**")
                
                # Hiển thị xác suất nếu có
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)[0]
                    probs_dict = dict(zip(model.classes_, probs))
                    
                    # Sắp xếp theo xác suất giảm dần
                    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    st.subheader("Xác suất các lớp:")
                    for cls, prob in sorted_probs:
                        st.write(f"- **{cls}**: {prob:.4f}")
                    
                    # Hiển thị biểu đồ xác suất
                    fig, ax = plt.subplots(figsize=(10, 6))
                    classes = [cls for cls, _ in sorted_probs]
                    probs = [prob for _, prob in sorted_probs]
                    sns.barplot(x=classes, y=probs, ax=ax)
                    plt.title('Xác suất các lớp')
                    plt.ylabel('Xác suất')
                    plt.xlabel('Lớp')
                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {str(e)}")
    else:
        st.info("Vui lòng huấn luyện mô hình trước khi thực hiện dự đoán. Hãy quay lại tab 'Tiền xử lý & Mô hình' hoặc 'Fine-tune & Tối ưu tham số' để huấn luyện mô hình.")
