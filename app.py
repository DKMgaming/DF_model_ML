import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from io import BytesIO
from math import atan2, degrees, radians, sin, cos, sqrt
import folium
from streamlit_folium import st_folium

# --- Hàm phụ ---
def calculate_azimuth(lat1, lon1, lat2, lon2):
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    x = sin(d_lon) * cos(lat2)
    y = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(d_lon)
    azimuth = (degrees(atan2(x, y)) + 360) % 360
    return azimuth

def simulate_signal_strength(dist_km, h, freq_mhz):
    path_loss = 32.45 + 20 * np.log10(dist_km + 0.1) + 20 * np.log10(freq_mhz + 1)
    return -30 - path_loss + 10 * np.log10(h + 1)

def calculate_destination(lat1, lon1, azimuth_deg, distance_km):
    R = 6371.0
    brng = radians(azimuth_deg)
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = np.arcsin(sin(lat1) * cos(distance_km / R) + cos(lat1) * sin(distance_km / R) * cos(brng))
    lon2 = lon1 + atan2(sin(brng) * sin(distance_km / R) * cos(lat1), cos(distance_km / R) - sin(lat1) * sin(lat2))
    return degrees(lat2), degrees(lon2)

def create_folium_map(data=None):
    if data is not None and not data.empty:
        m = folium.Map(location=[data['lat_receiver'].mean(), data['lon_receiver'].mean()], zoom_start=8, tiles="Stadia.StamenTonerLite")
    else:
        m = folium.Map(location=[16.0, 108.0], zoom_start=6, tiles="Stadia.StamenTonerLite")
    return m

def create_kml(results):
    kml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    kml += '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
    kml += '<Document>\n'
    kml += '<name>Dự đoán Tọa độ Nguồn Phát Xạ</name>\n'
    for index, result in enumerate(results):
        kml += '<Placemark>\n'
        kml += f'<name>Nguồn Phát Dự Đoán {index + 1}</name>\n'
        kml += '<Point>\n'
        kml += f'<coordinates>{result["lon_pred"]},{result["lat_pred"]},0</coordinates>\n'
        kml += '</Point>\n'
        kml += '<ExtendedData>\n'
        kml += f'<Data name="Lat_Receiver"><value>{result["lat_receiver"]}</value></Data>\n'
        kml += f'<Data name="Lon_Receiver"><value>{result["lon_receiver"]}</value></Data>\n'
        kml += f'<Data name="Predicted_Distance_km"><value>{result["predicted_distance_km"]:.2f}</value></Data>\n'
        kml += f'<Data name="Frequency_MHz"><value>{result["frequency"]}</value></Data>\n'
        kml += f'<Data name="Signal_Strength_dBm"><value>{result["signal_strength"]}</value></Data>\n'
        kml += '</ExtendedData>\n'
        kml += '</Placemark>\n'
        kml += '<Placemark>\n'
        kml += f'<name>Trạm Thu {index + 1}</name>\n'
        kml += '<Point>\n'
        kml += f'<coordinates>{result["lon_receiver"]},{result["lat_receiver"]},0</coordinates>\n'
        kml += '</Point>\n'
        kml += '</Placemark>\n'
    kml += '</Document>\n'
    kml += '</kml>\n'
    return kml

def create_single_kml(lat_rx, lon_rx, lat_pred, lon_pred, predicted_distance, frequency, signal_strength):
    kml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    kml += '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
    kml += '<Document>\n'
    kml += '<name>Dự đoán Tọa độ Nguồn Phát Xạ</name>\n'
    kml += '<Placemark>\n'
    kml += '<name>Nguồn Phát Dự Đoán</name>\n'
    kml += '<Point>\n'
    kml += f'<coordinates>{lon_pred},{lat_pred},0</coordinates>\n'
    kml += '</Point>\n'
    kml += '<ExtendedData>\n'
    kml += f'<Data name="Lat_Receiver"><value>{lat_rx}</value></Data>\n'
    kml += f'<Data name="Lon_Receiver"><value>{lon_rx}</value></Data>\n'
    kml += f'<Data name="Predicted_Distance_km"><value>{predicted_distance:.2f}</value></Data>\n'
    kml += f'<Data name="Frequency_MHz"><value>{frequency}</value></Data>\n'
    kml += f'<Data name="Signal_Strength_dBm"><value>{signal_strength}</value></Data>\n'
    kml += '</ExtendedData>\n'
    kml += '</Placemark>\n'
    kml += '<Placemark>\n'
    kml += '<name>Trạm Thu</name>\n'
    kml += '<Point>\n'
    kml += f'<coordinates>{lon_rx},{lat_rx},0</coordinates>\n'
    kml += '</Point>\n'
    kml += '</Placemark>\n'
    kml += '</Document>\n'
    kml += '</kml>\n'
    return kml

# --- Giao diện ---
st.set_page_config(layout="wide")
st.title("🔭 Dự đoán tọa độ nguồn phát xạ theo hướng định vị")

tab1, tab2 = st.tabs(["1. Huấn luyện mô hình", "2. Dự đoán tọa độ"])

# --- Tab 1: Huấn luyện ---
with tab1:
    st.subheader("📡 Huấn luyện mô hình với dữ liệu mô phỏng hoặc thực tế")

    option = st.radio("Chọn nguồn dữ liệu huấn luyện:", ("Sinh dữ liệu mô phỏng", "Tải file Excel dữ liệu thực tế"))

    df = None  # Đặt mặc định tránh lỗi NameError

    if option == "Sinh dữ liệu mô phỏng":
        if st.button("Huấn luyện mô hình từ dữ liệu mô phỏng"):
            st.info("Đang sinh dữ liệu mô phỏng...")
            np.random.seed(42)
            n_samples = 1000  # Tạo 1000 mẫu dữ liệu mô phỏng
            data = []
            for _ in range(n_samples):
                lat_tx = np.random.uniform(10.0, 21.0)
                lon_tx = np.random.uniform(105.0, 109.0)
                lat_rx = lat_tx + np.random.uniform(-0.05, 0.05)
                lon_rx = lon_tx + np.random.uniform(-0.05, 0.05)
                h_rx = np.random.uniform(5, 50)
                freq = np.random.uniform(400, 2600)

                azimuth = calculate_azimuth(lat_rx, lon_rx, lat_tx, lon_tx)
                distance = sqrt((lat_tx - lat_rx)**2 + (lon_tx - lon_rx)**2) * 111
                signal = simulate_signal_strength(distance, h_rx, freq)

                data.append({
                    "lat_receiver": lat_rx,
                    "lon_receiver": lon_rx,
                    "antenna_height": h_rx,
                    "azimuth": azimuth,
                    "frequency": freq,
                    "signal_strength": signal,
                    "distance_km": distance
                })

            df = pd.DataFrame(data)
            st.success("Dữ liệu mô phỏng đã được sinh thành công!")
            st.dataframe(df.head())
            towrite = BytesIO()
            df.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(
                label="📥 Tải dữ liệu mô phỏng (.xlsx)",
                data=towrite,
                file_name="simulation_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        uploaded_data = st.file_uploader("📂 Tải file Excel dữ liệu thực tế", type=["xlsx"])
        if uploaded_data:
            df = pd.read_excel(uploaded_data)
            st.success("Đã tải dữ liệu thực tế.")
            st.dataframe(df.head())
        else:
            st.info("Vui lòng tải file dữ liệu để huấn luyện.")

    if df is not None and st.button("🔧 Tiến hành huấn luyện mô hình"):
        try:
            st.info("Đang huấn luyện mô hình...")
            df['azimuth_sin'] = np.sin(np.radians(df['azimuth']))
            df['azimuth_cos'] = np.cos(np.radians(df['azimuth']))
            X = df[['lat_receiver', 'lon_receiver', 'antenna_height', 'signal_strength', 'frequency', 'azimuth_sin', 'azimuth_cos']]
            y = df[['distance_km']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8]
            }
            model = XGBRegressor(random_state=42)
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5, cv=3, random_state=42)
            st.info("Đang thực hiện RandomizedSearchCV để tìm tham số tối ưu...")
            random_search.fit(X_train, y_train.values.ravel())
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            st.success(f"Huấn luyện xong - MAE khoảng cách: {mae:.3f} km")
            st.success(f"RMSE: {rmse:.3f} km")
            st.success(f"R²: {r2:.3f}")
            buffer = BytesIO()
            joblib.dump(best_model, buffer)
            buffer.seek(0)
            st.download_button(
                label="📥 Tải mô hình huấn luyện (.joblib)",
                data=buffer,
                file_name="distance_model.joblib",
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")
            st.exception(e)

# --- Tab 2: Dự đoán ---
with tab2:
    st.subheader("📍 Dự đoán tọa độ nguồn phát xạ")

    uploaded_model = st.file_uploader("📂 Tải mô hình đã huấn luyện (.joblib)", type=["joblib"])
    if uploaded_model:
        model = joblib.load(uploaded_model)

        uploaded_excel = st.file_uploader("📄 Hoặc tải file Excel chứa thông tin các trạm thu", type=["xlsx"])

        if uploaded_excel:
            df_input = pd.read_excel(uploaded_excel)
            results = []

            if 'prediction_map' not in st.session_state:
                st.session_state['prediction_map'] = create_folium_map(df_input)
            else:
                new_center = [df_input['lat_receiver'].mean(), df_input['lon_receiver'].mean()]
                if st.session_state['prediction_map'].location != new_center:
                    st.session_state['prediction_map'] = create_folium_map(df_input)
                st.session_state['prediction_map']._children = {k: v for k, v in st.session_state['prediction_map']._children.items() if k.startswith('tile_layer') or k.startswith('crs')}

            for _, row in df_input.iterrows():
                az_sin = np.sin(np.radians(row['azimuth']))
                az_cos = np.cos(np.radians(row['azimuth']))
                X_input = np.array([[row['lat_receiver'], row['lon_receiver'], row['antenna_height'], row['signal_strength'], row['frequency'], az_sin, az_cos]])
                predicted_distance = model.predict(X_input)[0]
                predicted_distance = max(predicted_distance, 0.1)

                lat_pred, lon_pred = calculate_destination(row['lat_receiver'], row['lon_receiver'], row['azimuth'], predicted_distance)

                folium.Marker(
                    [lat_pred, lon_pred],
                    tooltip=f"Nguồn phát dự đoán\nTần số: {row['frequency']} MHz\nMức tín hiệu: {row['signal_strength']} dBm",
                    icon=folium.Icon(color='red')
                ).add_to(st.session_state['prediction_map'])

                folium.Marker([row['lat_receiver'], row['lon_receiver']], tooltip="Trạm thu", icon=folium.Icon(color='blue')).add_to(st.session_state['prediction_map'])
                folium.PolyLine(locations=[[row['lat_receiver'], row['lon_receiver']], [lat_pred, lon_pred]], color='green').add_to(st.session_state['prediction_map'])

                results.append({
                    "lat_receiver": row['lat_receiver'],
                    "lon_receiver": row['lon_receiver'],
                    "lat_pred": lat_pred,
                    "lon_pred": lon_pred,
                    "predicted_distance_km": predicted_distance,
                    "frequency": row['frequency'],
                    "signal_strength": row['signal_strength']
                })

            st.dataframe(pd.DataFrame(results))
            st_folium(st.session_state['prediction_map'], width=800, height=500, key="prediction_map")

            # Nút xuất file KML cho nhiều kết quả
            if results:
                kml_string = create_kml(results)
                b = BytesIO(kml_string.encode())
                st.download_button(
                    label="📤 Xuất file KML",
                    data=b,
                    file_name="predicted_locations.kml",
                    mime="application/vnd.google-earth.kml+xml"
                )

        else:
            with st.form("input_form"):
                lat_rx = st.number_input("Vĩ độ trạm thu", value=16.0)
                lon_rx = st.number_input("Kinh độ trạm thu", value=108.0)
                h_rx = st.number_input("Chiều cao anten (m)", value=30.0)
                signal = st.number_input("Mức tín hiệu thu (dBm)", value=-80.0)
                freq = st.number_input("Tần số (MHz)", value=900.0)
                azimuth = st.number_input("Góc phương vị (độ)", value=45.0)
                submitted = st.form_submit_button("🔍 Dự đoán tọa độ nguồn phát")

            if submitted:
                az_sin = np.sin(np.radians(azimuth))
                az_cos = np.cos(np.radians(azimuth))
                X_input = np.array([[lat_rx, lon_rx, h_rx, signal, freq, az_sin, az_cos]])
                predicted_distance = model.predict(X_input)[0]
                predicted_distance = max(predicted_distance, 0.1)

                lat_pred, lon_pred = calculate_destination(lat_rx, lon_rx, azimuth, predicted_distance)

                st.success("🎯 Tọa độ nguồn phát xạ dự đoán:")
                st.markdown(f"- **Vĩ độ**: `{lat_pred:.6f}`")
                st.markdown(f"- **Kinh độ**: `{lon_pred:.6f}`")
                st.markdown(f"- **Khoảng cách dự đoán**: `{predicted_distance:.2f} km`")

                if 'prediction_map' not in st.session_state:
                    st.session_state['prediction_map'] = create_folium_map(pd.DataFrame([{'lat_receiver': lat_rx, 'lon_receiver': lon_rx}]))
                else:
                    st.session_state['prediction_map'].location = [lat_rx, lon_rx]
                    st.session_state['prediction_map']._children = {k: v for k, v in st.session_state['prediction_map']._children.items() if k.startswith('tile_layer') or k.startswith('crs')}

                folium.Marker([lat_rx, lon_rx], tooltip="Trạm thu", icon=folium.Icon(color='blue')).add_to(st.session_state['prediction_map'])
                folium.Marker(
                    [lat_pred, lon_pred],
                    tooltip=f"Nguồn phát dự đoán\nTần số: {freq} MHz\nMức tín hiệu: {signal} dBm",
                    icon=folium.Icon(color='red')
                ).add_to(st.session_state['prediction_map'])
                folium.PolyLine(locations=[[lat_rx, lon_rx], [lat_pred, lon_pred]], color='green').add_to(st.session_state['prediction_map'])

                with st.container():
                    st_folium(st.session_state['prediction_map'], width=700, height=500, key="prediction_map_single")

                # Nút xuất file KML cho kết quả đơn lẻ
                kml_string_single = create_single_kml(lat_rx, lon_rx, lat_pred, lon_pred, predicted_distance, freq, signal)
                b_single = BytesIO(kml_string_single.encode())
                st.download_button(
                    label="📤 Xuất file KML",
                    data=b_single,
                    file_name="predicted_location.kml",
                    mime="application/vnd.google-earth.kml+xml"
                )

    else:
        st.info("Vui lòng tải mô hình đã huấn luyện để thực hiện dự đoán.")
