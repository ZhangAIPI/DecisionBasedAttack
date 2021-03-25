# Decision Based Attack 说明文档



## 1、在一个维度下两次优化，第二次的`V`如何计算？

如下图：

![image-20210325155342874](C:\Users\16844\AppData\Roaming\Typora\typora-user-images\image-20210325155342874.png)
$$
对V_{new}，有两个关系：\\
	（1）V_{new}垂直于U_{new}；\\
	（2）V_{new}和U、V在同一个平面内。\\
可以设未归一化的V_{new}=U+\alpha*V，那么有：\\
(U+\alpha V) \cdot U_{new}=0\\
解得V_{new}=U-\frac{U\cdot U_{new}}{V\cdot U_{new}} V\\
然后对其归一化:V_{new}=\frac{V_{new}}{||V_{new}||_2}
$$
