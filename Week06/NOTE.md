# 6-12 动态规划

## 递归代码模板

```python
def recursion(level, param):
    # terminator
    if level >= MAX_level:
        return
    # process current level logic
    process(level, param)
    # drill down
    self.recursion(level+1, newparam)
    # restore current states
```

## 分治代码模板

```python
def divide_conquer(problem, para1, para2, ...):
    # recursion terminator
    if problem == None:
        print result
        return
    # prepare data
    data = prepare_data(problem)
    # conquer subproblems
    subproblem = split_problem(problem, data)
    subresult1 = self.divide_conquer(subproblem[0], p1, ...)
    subresult2 = self.divide_conquer(subproblem[1], p1, ...)
    subresult3 = self.divide_conquer(subproblem[2], p1, ...)
    # process and generate the final result
    result = process_result(subresult1, subresult2, subresult3, ...)
```

**数学归纳法/寻找重复性**

**动态规划：分治+最优子结构**

## 关键点

1. 动态规划、递归、分治没有本质区别，关键看是否有最优子结构
2. 共性：找到重复子问题
3. 差异性：最优子结构、中途可以淘汰次优子结构

## 例题

1. 格子路径图：动态规划转移方程（例子中从终点到起点）

- 最优子结构
- 存储中间状态
- 递推公式，美其名曰 状态转移方程或递推方程
- 最开始赋值初始条件

2. 最长公共子串

- m+1,n+1行矩阵，方便初始计算
- 公共子串多转化为二维矩阵问题

## 动态规划小结

1. 打破思维惯性， 形成机器思维（自己变成机器）
2.  理解复杂逻辑的关键
3. 职业进阶的要点要领

## 实战例题

1. 爬楼梯变形： a. 爬1、2、3步；b.相邻步数不同；
2. 偷房子：一维数组不能解决问题，换二维

## 递归求解过程

1. 子问题
2. dp数组
3. dp方程