# 4-9 深度优先搜索和广度优先搜索

## 搜索和遍历

1.每个节点都要访问

2.每个节点仅访问一次

3.对节点的访问顺序

- 深度优先

- 广度优先
- 优先级优先

### 深度优先搜索代码模板

```python
### 二叉树
def dfs(node):
	if node in visited:
		return
	visited.add(node)
	### process current node here
	### ...logic here
	dfs(node.left)
	dfs(node.right)
### 多叉树
def dfs(node):
	if node in visited:
		return
	visited.add(node)
	### process current node here
	### ...logic here
	for children in node.childrens:
		if children not in visited:
			dfs(node.children)
```

### 广度优先搜索代码模板

```python
def dfs(graph, start, end):
    quene = []
    quene.append(start)
    visited.add(start)
    while quene:
        node = quene.pop()
        visited.add(node)
        process(node)
        nodes = generate_related_nodes(node)
        quene.push(nodes)
```

#### 小结

1.深度优先搜索可用递归/栈实现，时间复杂度为O(n)，空间复杂度为O(n)。

2.广度优先搜索可用队列实现（Python里collections.deque()），时间、空间复杂度与深度优先相同，适用于最短路径搜索。



# 4-10 贪心算法 Greedy

1.每一步都达到最优，从而达到全局最优

2.贪心VS动态规划，贪心不能回退，动态规划保留之前计算结果，可以回退

3.贪心算法可作为辅助算法，或解决一些不要求结果很精确的问题

4.贪心算法的关键在于确认问题的局部最优选择能够达到全局最优



# 4-11 二分查找

## 二分查找的前提

1.单调性

2.存在上下界

3.能够索引

### 二分查找代码模板

```python
left, right = 0, len(array) - 1
while left < right:
    mid = (left + right) / 2
    if array[mid] == target:
        break or return result
    elif array[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```

* 代码技巧
  - 五毒神掌
  - 审题 clarification（细节、边界条件、输入输出范围、特例情况）
  - 列举所有解法（给出时间复杂度、空间复杂度），选择最优解法，跟面试官探讨，在同意情况下继续
  - 进行coding
  - 给出相应测试样例， 测试的case一般都是那些，边界条件，错误样例，都能handle

### 牛顿迭代法

用直线代替曲线，利用一阶泰勒展开

#### 小结

1. 二分查找是逐次查找目标值在二分的哪份中
