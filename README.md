# 實務專題 - Neural Receiver (Neural RX)

## 📌 專案簡介
本專案為實務專題成果，主要探討 **Neural Receiver (Neural RX)** 的設計與應用。  
Neural RX 是一種結合深度學習與通訊系統的研究方向，透過神經網路實現接收端訊號處理，以提升無線通訊中在雜訊、干擾下的效能。  

本專題透過參考公開的 GitHub 專案，並在此基礎上進行改良與應用，完成了專題成果報告及實作程式。

---

## 📂 資料夾結構
```
實務專題/
│
├── neural_rx/                # Neural RX 相關程式碼與環境
│   ├── neural_rx/            # 主程式碼與 LICENSE
│   ├── .conda/               # 依賴套件與 Python 環境
│   └── ...
│
└── 專題成果報告.pdf           # 本專題的最終成果報告
```

---

## 🧪 專題說明
本專題的研究重點如下：
- 熟悉 **深度學習於通訊系統中的應用**，特別是接收端訊號處理。  
- 使用 **TensorFlow/Keras** 與 **Sionna** 工具進行建模與模擬。  
- 透過 **Ray-Tracing channel** 模型進行通道模擬，並結合神經網路架構做效能比較。  
- **設計多組實驗**，模擬不同通道條件（AWGN、RayTracing、多徑衰落），並與傳統接收器進行比較。  
- **測試網路結構與超參數**，例如層數、學習率、批次大小，並記錄對效能與收斂速度的影響。  
- 分析 **BER/BLER、收斂曲線、運算時間** 等指標，建立完整的評估流程。  

---

## 🚀 實作成果
- **程式實作**  
  - 參考 NVIDIA Sionna 以及相關 GitHub 專案程式，並依照專題需求進行調整。  
  - 建立神經網路接收端的模組，實現與通訊模擬系統的整合。  

- **實驗設計**  
  - **通道測試**：建立 AWGN、Rayleigh、以及多徑通道模型。  
  - **訊號參數**：測試不同的調變方式 (QPSK, 16-QAM) 與 SNR 範圍。  
  - **模型比較**：設計多種網路架構 (DNN, CNN, LSTM) 並比較表現差異。  
  - **超參數研究**：觀察不同學習率、批次大小對模型收斂的影響。  

- **測試成果**  
  - Neural RX 在 **低 SNR** 條件下 BER 顯著低於傳統接收器。  
  - 超參數調整結果顯示：較小的學習率與正規化能避免過擬合，加速收斂。  
  - 整體而言，Neural RX 在不同通道環境下均展現出比傳統接收器更佳的魯棒性。  

- **專題報告**  
  在 `專題成果報告.pdf` 中，詳細呈現了研究動機、方法、實驗設計、模擬結果與結論。  

- **個人貢獻**  
  - 修改並整合原始程式碼，使其能支援專題所需功能。  
  - 規劃多組實驗設計，並完成測試與數據分析。  
  - 撰寫專題報告，彙整研究成果與心得。  

---

## 📜 版權聲明
以下為 `neural_rx/neural_rx/LICENSE.txt` 之完整內容：  

```
NVIDIA License

1. Definitions

“Licensor” means any person or entity that distributes its Work.
“Work” means (a) the original work of authorship made available under this license, which may include software, documentation, or other files, and (b) any additions to or derivative works  thereof  that are made available under this license.
The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under U.S. copyright law; provided, however, that for the purposes of this license, derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.
Works are “made available” under this license by including in or with the Work either (a) a copyright notice referencing the applicability of this license to the Work, or (b) a copy of this license.

2. License Grant

    2.1 Copyright Grant. Subject to the terms and conditions of this license, each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free, copyright license to use, reproduce, prepare derivative works of, publicly display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.

3. Limitations

    3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this license, (b) you include a complete copy of this license with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.

    3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms. Notwithstanding Your Terms, this license (including the redistribution requirements in Section 3.1) will continue to apply to the Work itself.

    3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation and its affiliates may use the Work and any derivative works commercially. As used herein, “non-commercially” means for research or evaluation purposes only.

    3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your rights under this license from such Licensor (including the grant in Section 2.1) will terminate immediately.

    3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or trademarks, except as necessary to reproduce the notices described in this license.

    3.6 Termination. If you violate any term of this license, then your rights under this license (including the grant in Section 2.1) will terminate immediately.

4. Disclaimer of Warranty.

THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE.

5. Limitation of Liability.

EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

```