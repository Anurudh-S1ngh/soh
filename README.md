# SoH estimation of Lithium-ion battery
- This project is designed to predict State of health (SoH) for identifying remaining useful life of Li-ion batteries.



## Process  
### **1**. Calculating and Visualizing SoH with 7 Li-ion battery datasets | [Code](https://github.com/Anurudh-S1ngh/soh/blob/main/1_Calculation_and_Visulaliztion_of_SoH/Calculation_and_Visualization_of_SoH.ipynb)

<div align="center">
<img src="https://github.com/OH-Seoyoung/SoH_estimation_of_Lithium-ion_battery/blob/master/1_Calculation_and_Visulaliztion_of_SoH/fig/SoH_B05.jpg?raw=True" width="48%">
<img src="https://github.com/OH-Seoyoung/SoH_estimation_of_Lithium-ion_battery/blob/master/1_Calculation_and_Visulaliztion_of_SoH/fig/SoH_B47.jpg?raw=True" width="48%"> <br>
</div>  

### **2**. Eliminating outliers with quantile | [Code](https://github.com/Anurudh-S1ngh/soh/blob/main/2_Elimination_of_outliers/Calculation_and_Visualization_of_refined_SoH.ipynb) 

<div align="center">
<img src="https://github.com/OH-Seoyoung/SoH_estimation_of_Lithium-ion_battery/blob/master/2_Elimination_of_outliers/fig/A_group.jpg?raw=True" width="48%">
<img src="https://github.com/OH-Seoyoung/SoH_estimation_of_Lithium-ion_battery/blob/master/2_Elimination_of_outliers/fig/C_group.jpg?raw=True" width="48%"> <br>
</div>  


### **4**. Long Short Term Memory | [Code](https://github.com/Anurudh-S1ngh/soh/blob/main/4_LSTM_with_SoH/SoH_estimation_with_LSTM.ipynb)
- Start at **50%** Cycle
<div align="center">
  <div style="display: inline-block; text-align: center;">
    <h2>B07 50%</h2>
    <img src="https://github.com/user-attachments/assets/fc6ba164-66e8-4d49-9575-3c8d93479102" width="48%">
  </div>
  <div style="display: inline-block; text-align: center;">
    <h2>B07 50%</h2>
    <img src="https://github.com/user-attachments/assets/4c81143b-bd10-4e58-8584-4bd64ba3d4de" width="48%">
  </div>
  <div>
    <h2>B18 50%</h2>
    <img src="https://github.com/user-attachments/assets/ffcbf183-80d9-41ea-be6c-138c543218bb" width = "48%">
  </div>
</div>


</div>  

- Start at **70%** Cycle
<div align="center">
<img src="https://github.com/OH-Seoyoung/SoH_estimation_of_Lithium-ion_battery/blob/master/4_LSTM_with_SoH/70%25/fig/B05_LSTM.jpg?raw=True" width="48%">
<img src="https://github.com/OH-Seoyoung/SoH_estimation_of_Lithium-ion_battery/blob/master/4_LSTM_with_SoH/70%25/fig/B47_LSTM.jpg?raw=True" width="48%"> <br>
  <h1>B05 70%</h1>
  <img src="https://github.com/user-attachments/assets/a85bd139-6aa5-4e9d-bce4-9c5352c57842" width="48%">
  <br>
  <h1>B07 70%</h1>
  <img src="https://github.com/user-attachments/assets/4399f485-8c2c-40ff-88d8-1d77d4e689ab" width="48%">
</div>  

## Results


## Dataset  

```
[1] NASA Prognostic Center: Experiments on Li-ion Batteries, https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/ 
```
