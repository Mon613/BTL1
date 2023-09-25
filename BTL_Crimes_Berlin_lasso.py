import numpy as np # cung cấp các hàm và lớp để xử lý các mảng và ma trận.
import pandas as pd #  cung cấp các hàm và lớp để xử lý dữ liệu dạng bảng
from sklearn.linear_model import LinearRegression, Ridge, Lasso # cấp các mô hình hồi quy tuyến tính
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # cung cấp các hàm để đánh giá mô hình sử dụng để tính toán các chỉ số R-squared, NSE, MAE và RMSE
from sklearn.model_selection import train_test_split, KFold #cung cấp các kỹ thuật để chia dữ liệu và đánh giá mô hình. sử dụng để chia dữ liệu thành hai tập: tập huấn luyện và tập kiểm tra
from hydroeval import evaluator, nse #thư viện này được sử dụng để tính toán chỉ số NSE theo tiêu chí Nash-Sutcliffe efficiency

### Hàm đánh giá NSE
def nse(targets,predictions):
    return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(predictions))**2)))

### Hàm tím train error, validation error
def train_error(targets,predictions):
    return np.mean(np.square(np.array(predictions)-np.array(targets)))
# Hàm sử dụng các hàm NumPy để chuyển đổi tập dữ liệu mục tiêu và tập dữ liệu dự đoán thành các mảng NumPy, 
# tính bình phương của sự khác biệt giữa hai mảng, và tính trung bình của các giá trị bình phương

def lasso():
    lassoReg = Lasso()
    lassoReg.fit(xTrain, yTrain)# train mô hình
    predict = np.array(lassoReg.predict(xTest))# dự đoán trên tập test
    train_set = lassoReg.predict(xTrain)# dự đoán trên tập train
    return {'Title': 'Lasso',
    'R2 score': r2_score(yTest, predict),# Tính điểm r2 trên tập test
    'Score NSE': nse(yTest, predict),
    'Score NSE by hydroeval': evaluator(nse, predict, yTest),
    'Score MAE': mean_absolute_error(yTest, predict),
    'Score RMSE': mean_squared_error(yTest, predict)**0.5,
    'Test error': train_error(yTest.tolist(), predict.tolist()),
    'Train error': train_error(yTrain.tolist(), train_set.tolist())}

data = pd.read_csv('./BTL1/solieu1.csv') ## Load data

dTrain, dTest = train_test_split(data, test_size=0.3, shuffle=False) ## Split data
xTrain, yTrain = np.array(dTrain.iloc[:,:11]), np.array(dTrain.iloc[:,11]) ## Data train model
xTest, yTest = np.array(dTest.iloc[:,:11]), np.array(dTest.iloc[:,11])

ll = lasso() 
 ## Sort model by key (R2 score) descending

### Print
for j in ll:
    print(j, ': ', ll[j], sep='')


