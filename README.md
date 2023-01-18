# Robust Energy Management System

💡 energy management systems(EMS) is becoming more complex as fossil fuel-based energy production decreases and renewable energy increases. Especially for net-zero energy communities and isolated microgrids, which receive a limited amount of energy from the external grid, the EMS uncertainty is even greater.

 To address the uncertainties and avoid such failures, robust energy management methods have been proposed based on model-based and model-free approaches. model-free methods have been proposed by using safe reinforcement learning (RL) which implicitly learn uncertainties to avoid unsafe situations. However, such implicit learning cannot catch up with the accuracy of state-of-the-art forecasting methods that fully exploit real-world context information via deep learning. Hence, it can be expected better performance if such accurate forecasts can be seamlessly exploited in energy management along with safe RL 
 
 In this letter, we propose an EMS algorithm robust to inconsistent energy supply based on safe RL. Specifically, it minimizes energy costs while avoiding failures to satisfy energy demands.

## Cumulative Failure
- Train = 2017-01-01 ~ 2017-01-30
- Test = 2017-01-31 ~ 2017-04-30

![다운로드](https://user-images.githubusercontent.com/79639187/212813389-200feb7b-49fe-42c0-a5b2-c3fed5721a28.png)

## Test Period EMS Status

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sATG9i3L3rPnJGTI1ObhOENNJjhibTaF#scrollTo=zTINcb41GUtP)
