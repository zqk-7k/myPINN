说明：
本仓库使用python代码模拟引力波放大率曲线f(y,omega)，涉及的其中y取值为1，5，10，40；omega取值范围为[0.01,0.1]。具体过程可见文件Gravitational_Wave_Lensing.pdf。

amplification_test.py
python自带公式模拟放大率曲线:
![放大率](https://github.com/user-attachments/assets/dfbffcee-f252-4f31-afce-ade4a7282bf2)

PINN_torch2.py
仅用数据损失进行训练，未引入任何物理损失，拟合结果尚可。
![image](https://github.com/user-attachments/assets/c689aa1c-a3ea-45e5-bd04-95e168ef1f0b)


pinn_test3.py
数据损失加物理损失，其中还有边界约束条件。
![image](https://github.com/user-attachments/assets/8a5799a7-cf9d-4f90-8656-b04fbcb8b0b6)

PINN_Kummer_Physics_BC_y10.py
固定y=10，模拟结果，数据损失，添加了1的边界值约束，添加了尾部边界约束。
![image](https://github.com/user-attachments/assets/2a2fb92b-4794-448f-936e-f9c11d4e7c2a)
