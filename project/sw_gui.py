import sys
import os
import cv2
import glob
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout,
                                QLabel, QGroupBox, QTableWidget,
                                QTableWidgetItem, QAction, QMainWindow, QWidget, QSizePolicy, 
                                QFileDialog, QScrollArea, QListWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap

from PIL import Image
import numpy as np
import pandas as pd

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("bf-logo.png"))
        self.initUI()
        

    def initUI(self):
        # Menu bar setup
        load_image_action = QAction(QIcon('edit.png'), '폴더 선택', self)
        load_image_action.setStatusTip('폴더 선택')
        load_image_action.triggered.connect(self.load_images_from_folder)

        self.statusBar()

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('&설정')
        filemenu.addAction(load_image_action)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top layout (Image view + Result list)
        top_layout = QHBoxLayout()

        # Image scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedSize(720, 540)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #606A71;")
        scroll_area.setWidget(self.image_label)

        # Result list (Deep Learning Results)
        self.result_groupbox = QGroupBox("이미지 파일 목록")
        self.result_groupbox.setFixedSize(270, 540)
        self.result_list = QListWidget()
        self.result_list.itemClicked.connect(self.load_image)
        result_layout = QVBoxLayout(self.result_groupbox)
        result_layout.addWidget(self.result_list)

        top_layout.addWidget(scroll_area)
        top_layout.addWidget(self.result_groupbox)
        main_layout.addLayout(top_layout)

        # Bottom layout (Results)
        bottom_layout = QVBoxLayout()
        settings_groupbox = QGroupBox("불량검출 결과")
        settings_groupbox.setFixedSize(1005, 140)
        settings_layout = QVBoxLayout(settings_groupbox)

        # 상태 메시지 레이블 초기화
        self.status_label = QLabel(" ")  # 초기값 설정
        self.status_label.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(self.status_label)  # 설정 레이아웃에 추가

        self.table = QTableWidget(3, 3)

        self.table.setHorizontalHeaderLabels(['불량분류', '정확도', '위치(x1, y1, x2, y2)'])
        self.table.verticalHeader().setVisible(False)  # 세로 헤더 숨기기

        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # 반응형
        
        # 열 너비 설정
        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 300)  # 좌표 정보 열 너비 설정

        # 초기값으로 "-"를 테이블의 모든 셀에 설정 (3행, 3열)
        for row in range(3):  # 0, 1, 2 인덱스에 대해
            for col in range(3):  # 0, 1, 2 인덱스에 대해
                self.table.setItem(row, col, QTableWidgetItem("-"))

        settings_layout.addWidget(self.table)
        bottom_layout.addWidget(settings_groupbox)
        main_layout.addLayout(bottom_layout)

        self.setWindowTitle('PCB 불량 검출 딥러닝 프로그램') 
        self.setFixedSize(1024, 768)

    # ================================================================================================================= #
    # ============================================= qt design end ===================================================== #
    # ================================================================================================================= #

    def load_images_from_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, '폴더 선택', r'C:\python_dev\sw_copyright\cropped_img')
        if folder_path:
            # 폴더에서 이미지 파일만 선택
            image_files = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))
            self.result_list.clear()  # 기존 목록 초기화
            self.image_file_paths = image_files

            # 이미지 목록을 리스트에 추가
            for image_file in image_files:
                image_name = os.path.basename(image_file)
                self.result_list.addItem(image_name)

    def load_image(self, item):
        selected_image = item.text()
        folder_path = os.path.dirname(self.image_file_paths[0])  # 이미지 경로에서 폴더 경로 추출
        image_path = os.path.join(folder_path, selected_image)

        # 경로에서 백슬래시를 슬래시로 변환
        image_path = image_path.replace('\\', '/') 
        
        if os.path.exists(image_path):
            try:
                # PIL로 이미지 로드
                img = Image.open(image_path)

                # 이미지를 QImage로 변환
                img = img.convert('RGB')  # 3채널 RGB로 변환
                data = img.tobytes("raw", "RGB")
                qimage = QImage(data, img.width, img.height, QImage.Format_RGB888)

                # if qimage.isNull():
                #     raise Exception("QImage failed to load the image from PIL data")
                
                # QPixmap 생성 및 크기 조절
                pixmap = QPixmap.fromImage(qimage)
                scaled_pixmap = pixmap.scaled(720, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)

                # 상태 메시지 표시
                if self.status_label is not None:
                    self.status_label.setText("불량 탐색 중...")
                # 이미지 전처리 & 딥러닝 모델 적용
                processed_image = self.preprocess_image_for_model(image_path)
                QApplication.processEvents()  # UI 업데이트를 위한 이벤트 처리
                result = self.apply_deep_learning_model(processed_image)

                # 상태 메시지 제거
                self.status_label.setText("")

                # 딥러닝 결과 이미지 저장 경로
                runs_dir = "runs/detect"
                detect_folders = glob.glob(os.path.join(runs_dir, 'predict*'))
                if detect_folders:  # detect 폴더가 존재하는지 확인
                    latest_detect_folder = max(detect_folders, key=os.path.getctime)
                    result_images = glob.glob(os.path.join(latest_detect_folder, '*'))

                    if result_images:
                        result_image_path = result_images[0] # 이미지 하나만 가져오기
                        # print(f"Loading image from: {result_image_path}")

                        # 결과 이미지 표시
                        result_img = Image.open(result_image_path)
                        result_img = result_img.convert('RGB')
                        data_result = result_img.tobytes("raw", "RGB")
                        qimage_result = QImage(data_result, result_img.width, result_img.height, QImage.Format_RGB888)
                        
                        # QPixmap 생성 및 크기 조절
                        pixmap_result = QPixmap.fromImage(qimage_result)
                        scaled_pixmap_result = pixmap_result.scaled(720, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.image_label.setPixmap(scaled_pixmap_result)  # 결과 이미지를 QLabel에 표시

                        print(f"Result image loaded successfully: {result_image_path}")
                    else:
                        print("No images found in the latest detection folder.")

                # 딥러닝 결과(불량 종류, precision, 좌표)를 테이블에 업데이트
                self.update_result_table(result)

            except Exception as e:
                print(f"Error loading image: {e}")
    
    # 이미지 전처리 함수
    def preprocess_image_for_model(self, image_path):
        # OpenCV로 이미지 로드
        img = cv2.imread(image_path)

        # 이미지가 None이면 로드 실패
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 이미지 크기 출력
        print(f"Loaded image shape: {img.shape}")
        
        # 딥러닝 모델을 위한 이미지 전처리 (크기 조정 등)
        # dsize가 설정되어 있는지 확인 필요
        resized_img = cv2.resize(img, (640, 640))  # 여기에 크기 조정
        return resized_img

    # 딥러닝 적용
    def apply_deep_learning_model(self, processed_image):
        from ultralytics import RTDETR
        
        model = RTDETR(r"C:\yolonasdetr\model\sw_test01_rtdetr_l.pt")  # load a custom model
        conf_threshold = 0.6 # Set confidence threshold

        # prediction = model.predict(processed_image)  # 예측 실행
        results = model(processed_image, save=True, conf=conf_threshold)  # predict on an image
        
        detections = []  # 결과 저장 리스트

        # 예측 결과 처리
        for r in results:
            boxes = r.boxes  # 박스 객체들
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 박스 좌표
                conf = box.conf[0].item()  # 신뢰도 점수
                cls = int(box.cls[0].item())  # 클래스
                detections.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': conf,
                    'class': cls
                })

        # 결과를 DataFrame으로 변환
        df = pd.DataFrame(detections)
        
        '''
        # CSV로 저장
        output_dir = r"C:\yolonasdetr\test_img"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "detections.csv")
        df.to_csv(csv_path, index=False)
        print(f"Detection coordinates saved to {csv_path}")
        '''

        # 딥러닝 모델 예측 결과 반환
        return detections  # 반환할 값은 리스트 형태
    
    # 딥러닝 좌표를 테이블에 업데이트하는 함수
    def update_result_table(self, detections):
        num_detections = len(detections)
        if num_detections == 0:
            # 검출된 객체가 없을 경우
            self.table.setRowCount(1)  # 한 개의 행을 추가
            self.table.setItem(0, 0, QTableWidgetItem("검출된 불량이 없습니다."))
            self.table.setItem(0, 1, QTableWidgetItem(""))
            self.table.setItem(0, 2, QTableWidgetItem(""))
            self.table.setColumnWidth(0, 300)
            self.table.setColumnWidth(1, 0)
            self.table.setColumnWidth(2, 0)
        else:
            # 검출된 객체가 있을 경우
            self.table.setRowCount(num_detections)  # 테이블의 행 수를 예측된 객체 수로 설정

            # 테이블 헤더 설정 (열: 클래스, 신뢰도, 좌표)
            self.table.setHorizontalHeaderLabels(['불량분류', '정확도', '위치(x1, y1, x2, y2)'])

            # 예측된 각 객체의 정보를 테이블에 추가
            for i, detection in enumerate(detections):
                # 불량 int -> text
                if detection['class'] == 0:
                    class_name = "Ball Damage"
                elif detection['class'] == 1:
                    class_name = "Another Damage"  # class가 1일 때의 메시지
                else:
                    class_name = f"불량 {detection['class']}"  # 그 외의 경우
                confidence = f"{detection['confidence']:.2f}"
                coordinates = f"({detection['x1']:.1f}, {detection['y1']:.1f}, {detection['x2']:.1f}, {detection['y2']:.1f})"

                # 클래스 정보, 신뢰도, 좌표를 테이블에 추가
                self.table.setItem(i, 0, QTableWidgetItem(class_name))  # 클래스 정보
                self.table.setItem(i, 1, QTableWidgetItem(confidence))  # 신뢰도
                self.table.setItem(i, 2, QTableWidgetItem(coordinates))  # 좌표 정보

            self.table.setColumnWidth(0, 120)  # 좌표 정보 열 너비 설정
            self.table.setColumnWidth(1, 120)  # 좌표 정보 열 너비 설정
            self.table.setColumnWidth(2, 300)  # 좌표 정보 열 너비 설정

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())