Epoch：整个训练集被完整遍历一轮

Batch size:一次参数更新时使用的样本数量

Iteration/Step：模型处理一个Batch并完成一次参数更新

Iterations=总样本数/Batch size

Batch size太小，GPU利用率低，训练慢 Batch size越大，每个epoch越快，但是大到现存不够，训练会直接报错

显存占用与batch size’成正相关

Batch size影响梯度噪声（小batch size噪声大，有时能帮助找到更平坦泛化好的解）

影响收敛稳定性和泛化能力  （大batch size梯度更准确稳定，训练曲线更平滑，但也容易收敛较差的解）
