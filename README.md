# 神经网络手撸
使用pandas和numpy自己创建神经网络
代码完成个了一共二分类任务，具体内容为[x,y,z]这样一个三维数据，如果x+y+z大于15，那么分类为1，否则分类为0.
目前只完成了一小部分，可能分类准确率并不是那么高我会陆续调整优化网络，项目是自己随便想出来的，神经网络架构和
优化算法都是从吴恩达老师的网课上学的，然后自己代码实现，网站贴这里了
https://www.bilibili.com/video/BV1FT4y1E74V?p=71&spm_id_from=pageDriver
目前完成的功能：
1.数据集划分，归一化输入，神经网络模型的自定义（层数与节点数）
2.前向后向传播
3.批量梯度下降，动量梯度下降，PMSprop，Adam等优化算法
