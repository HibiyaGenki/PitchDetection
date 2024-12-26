import numpy as np
import cv2

def optical_flow(cap_url):
    cap = cv2.VideoCapture(cap_url)
    # Shi-Tomasiのコーナー検出パラメータ
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

    # Lucas-Kanade法のパラメータ
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # ランダムに色を１００個生成（値0～255の範囲で100行3列のランダムなndarrayを生成）
    color = np.random.randint(0, 255, (100, 3))
    end_flag, frame = cap.read()
    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    feature_prev = cv2.goodFeaturesToTrack(gray_prev, mask = None, **feature_params)
    mask = np.zeros_like(frame)

    while(end_flag):
        # グレースケールに変換
        gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # オプティカルフロー検出
        feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, feature_prev, None, **lk_params)

        # オプティカルフローを検出した特徴点を選別（0：検出せず、1：検出した）
        good_prev = feature_prev[status == 1]
        good_next = feature_next[status == 1]

        # オプティカルフローを描画
        for i, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
            prev_x, prev_y = prev_point.ravel()
            next_x, next_y = next_point.ravel()
            prev_x, prev_y = int(prev_x), int(prev_y)
            next_x, next_y = int(next_x), int(next_y)

            mask = cv2.line(mask, (next_x, next_y), (prev_x, prev_y), color[i].tolist(), 2)
            frame = cv2.circle(frame, (next_x, next_y), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        # ウィンドウに表示
        cv2.imshow('window', img)

        # ESCキー押下で終了
        if cv2.waitKey(30) & 0xff == 27:
            break
        gray_prev = gray_next.copy()
        feature_prev = good_next.reshape(-1, 1, 2)
        end_flag, frame = cap.read()
    
    cv2.destroyAllWindows()
    cap.release()
optical_flow('/Users/hibiyagenki/Downloads/4b_vsWASEDA.mp4')
