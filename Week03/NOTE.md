# 3-7 泛型递归 树的递归

**树的面试题解法一般都是递归**

<img src="C:\Users\HBQ\AppData\Roaming\Typora\typora-user-images\image-20200712105912253.png" alt="image-20200712105912253" style="zoom:50%;" />

## 代码模板

```python
def recursion(level, data):
    # recursion terminor
    if level >= MAX_level:
        return
    # process current logic
    process(level, data)
    # drill down
    self.recursion(level + 1, data)
    # reserve the current level state if needed
```

## 思维要点

1. 不要人肉递归
2. 找到最近最简方法，将其拆解为可重复子问题
3. 数学归纳法思想

## 习题

待完成

# 3-8 分治(divide&conquer)  回溯

<img src="C:\Users\HBQ\AppData\Roaming\Typora\typora-user-images\image-20200712110752013.png" alt="image-20200712110752013" style="zoom:50%;" />

<img src="C:\Users\HBQ\AppData\Roaming\Typora\typora-user-images\image-20200712112610830.png" alt="image-20200712112610830" style="zoom:50%;" />

<img src="C:\Users\HBQ\AppData\Roaming\Typora\typora-user-images\image-20200712112651715.png" alt="image-20200712112651715" style="zoom:50%;" />

## 习题

待重做

