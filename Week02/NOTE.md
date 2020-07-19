# 2-5 哈希表 映射 集合

## 概述

1. 哈希表又称散列表，通过关键码值进行访问的数据结构
2. 关键码值映射到表中的位置来加快访问速度
3. 映射函数称为散列函数（hash function）， 存放记录的数组称为哈希表或散列表

![image-20200711100930581](C:\Users\HBQ\AppData\Roaming\Typora\typora-user-images\image-20200711100930581.png)

## 习题

待重做

# 2-6-1 树 二叉树 二叉搜索树

## 知识点

1. 二维数据结构
2. 根 子节点 父节点 兄弟节点 子树 
3. 二叉树是有两个分叉的数
4. 树是特殊结构的图、链表是特殊结构的树

## 前序遍历

根->左->右

```python
def preorder(self, x):
    self.travel_path(x.val)
    self.preorder(x.left)
    self.preorder(x.right)
```

## 中序遍历

左->根->右

```python
def inorder(self, x):
    self.inorder(x.left)
    self.travel_path(x.val)
    self.inorder(x.right)
```

## 后序遍历

左->右->根

```python
def postorder(self, x):
    self.postorder(x.left)
    self.postorder(x.right)
    self.travel_path(x.val)
```

## 二叉搜索树

### 概述

1. 二叉搜索树又称有序二叉树、搜索二叉树、二叉查找树
2. 指一棵空树或具有以下性质的二叉树：

- 左子树上所有节点的值均小于其根节点的值
- 右子树上所有节点的值均大于其根节点的值
- 以此类推，左右子树又分别称为二叉搜索树
- 常见操作有：查询、插入新节点、删除

**——中序遍历：升序排列**

	# 习题

1. 中序遍历

```python
"""
递归
"""
res = []
recursion(root, res)
return res
def recursion(root, res):
    if root.left != None:
        recursion(root.left, res)
    res.append(root.val)
    if root.right != None:
        recursion(root.right, res)
```

```python
"""
迭代
"""
cur = root
stack = []
ans = []
while cur != None or len(stack) != 0:
    while cur != None:
        stack.append(cur)
        cur = cur.left
    cur = stack.pop()
    ans.append(cur.val)
    cur = cur.right
return ans
```

2. 前序遍历

```python
"""
递归
"""
if not root:
    return []
ans = []
recursion(root, ans)
return []
def recursion(root, ans):
    ans.append(root.val)
    if root.left != None:
        recursion(root.left, ans)
    if root.right != None:
        recursion(root.right, ans)
```

```python
"""
迭代
"""
ans = []
cur = root
stack = []
while cur != None or len(stack) != 0:
    while cur != None:
        stack.append(cur)
        ans.append(cur.val)
        cur = cur.right
    cur = stack.pop()
    cur = stack.right
return ans
```

3. 后序遍历

```python
"""
递归
"""
ans = []
recursion(root, ans)
return ans
def recursion(root, ans):
    if root.left != None:
        recursion(root.left, ans)
    if root.right != None:
        recursion(root.right, ans)
    ans.append(root.val)
```

```python
"""
迭代(前序遍历反转)
"""
ans = []
stack = []
cur = root
while cur != None and len(stack) != 0:
    while cur != None:
        stack.append(cur)
        ans.append(cur.val)
        cur = cur.right
    cur = stack.pop()
    cur = cur.left
return ans[::-1]
```

```python
## 倒序说明
list的[]中有三个参数，用冒号分割
list[param1:param2:param3]

param1，相当于start_index，可以为空，默认是0
param2，相当于end_index，可以为空，默认是list.size
param3，步长，默认为1。步长为-1时，返回倒序原序列
```

```python
## 后序遍历（多子树）
def postorder(self, root: 'Node') -> List[int]:
    if root:
        output = []
        stack = [root]
        while len(stack) > 0:
            current = stack.pop()
            output.append(current.val)
            for c in current.children:
                stack.append(c)
                return output[::-1]
    else:
        return []
```

# 2-6-2 图

## 图的属性和分类

1. Graph(V, E)
2. V-Vertex，点

- 度 入度&出度
- 点与点连通与否

3. E-edge， 边

- 有向边和无向边
- 权重（边长）

4. 图的表示和分类

- 邻接矩阵&链表
- 无向无权图/有向无权图/无向有权图

## 基于图的相关算法

1. 深度优先搜索

```python
visited = set()
def dfs(node, visited):
    if node in visited:
        return
    visited.add(node)
    for children in node.childrens():
        if children not in visited:
            dfs(children, visited)    
```

2. 广度优先搜素

```python
visited = set()
def bfs(node, visited):
    quene = [node]
    for i in range(len(quene)):
        cur = quene.popleft()
        visited.add(cur)
        for children in node.childrens():
            quene.add(children)            
```

# 2-6-3 堆(heap)、二叉堆(binary heap)

## 概述

1. 可以迅速找到一推数中最大或最小值的数据结构
2. 将根节点最大的叫做大顶推或大根堆、根节点最小的叫做小顶堆或小根堆
3. 常见的堆有二叉堆、斐波那契堆等
4. 假设是一大根堆，常见操作有：

- find-max O(1)
- delete-max O(logN)
- insert (creat) O(logN) (O())

5. 不同实现的比较

## 二叉搜索堆

1. 通过一棵完全二叉树实现
2. 是一棵完全树
3. 树中任意节点的值总是大于其子节点的值
4. 二叉堆一般通过数组实现
5. 假设第一个元素索引为0的话

- 索引为i的左孩子的索引为2*i+1
- 索引为i的右孩子的索引为2*i+2
- 索引为i的父节点的索引为floor((i-1)/2)

### 插入操作

1. 新元素一律到队尾
2. 一直向上调整堆的结构（直到根部）
3. heapifyup

## delect 堆max操作

1. 将堆尾元素替换到顶部
2. 向下调整堆结构（直到底部）
3. heapifydown

**二叉堆（优先队列priority_quene）是堆的一张常见和简单的实现，但并不是最优的实现**

### 习题

待重做

1. 最小的k个数
2. 出现频率最高的k个数
3. 滑动窗口