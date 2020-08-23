# 一、时间空间复杂度

## 1.常见数据结构操作时间复杂度

|                | A    | v    | e    | g    | W    | o    | r    | t    | Space Complexity |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---------------- |
|                | 访问 | 查询 | 插入 | 删除 | 访问 | 查询 | 插入 | 删除 |                  |
| Array          | 1    | n    | n    | n    | 1    | n    | n    | n    | n                |
| Stack          | n    | n    | 1    | 1    | n    | n    | 1    | 1    | n                |
| Quene          | n    | n    | 1    | 1    | n    | n    | 1    | 1    | n                |
| Linked List    | n    | n    | 1    | 1    | n    | n    | 1    | 1    | n                |
| Skip Table     | logn | logn | logn | logn | n    | n    | n    | n    | n                |
| Hash map       |      | 1    | 1    | 1    |      | n    | n    | n    | n                |
| Bi Search Tree | logn | logn | logn | logn | n    | n    | n    | n    | n                |
| AVL Tree       | logn | logn | logn | logn | logn | logn | logn | logn | n                |
| Red-Black Tree | logn | logn | logn | logn | logn | logn | logn | logn | n                |
| Trie           | n    | n    | n    | n    | n    | n    | n    | n    | n*m              |
| Disjoint       | n    | n    | n    | n    | n    | n    | n    | n    | n\               |

## 2.常见时间复杂度级别

O(1)	常数时间复杂度 Constant Complexity

O(logN)	对数时间复杂度 Logarithmic Complexity

O(N)	线性时间复杂度 Linear Complexity

O(NlogN)	线性-对数 Linear-Logarithmic Complexity

O(N^2)	平方 N square Complexity

O(N^3)	立方 N cubic Complexity

O(2^N)	指数 Exponential Growth

O(N!)	阶乘 Factorial

# 二、算法模板

## 1.树的遍历

```python
# 前序遍历
def preorder(root):
    if root:
        self.travel_path.append(root.val)
        self.preorder(root.left)
        self.preorder(root.right)
    
# 中序遍历
def inorder(root):
    if root:
        self.inorder(root.left)
        self.travel_path.append(root.val)
        self.inorder(root.right)
# 中序遍历-栈方法
def inorder(root):
    res = []
    stack = []
    cur_node = root
    while cur_node != None or len(stack) != 0:
        while cur_node != None:
            stack.append(cur_node)
            cur_node = cur_node.left
        cur_node = stack.pop()
        res.append(cur_node.val)
        cur_node = cur_node.right
# 后序遍历
def postorder(root):
    if root:
        self.postorder(root.left)
        self.postorder(root.right)
        self.travel_path.append(root.val)        
```

## 2.递归

```python
# 递归模板
def recursion(level, param1, param2...):
	# recursion teminator
    if level >= Max_Level:
        process_result
    # process logic in current level
    process(level, param1, param2...)
    # drill down
    recursion(level + 1, p1, p2...)
    # reverse the current status if needes
    process_again(level, param1, param2)
```

## 3.深度优先搜索+广度优先搜索

```python
# 深度优先搜索模板
## 递归写法
def dfs(node, visited):
    # recursion teminator
    if node in visited:
        return
    # process current logic 
    visited.add(node)
    # drill down
    for next_node in node.children:
        if next_node not in visited:
            dfs(next_node, visited)
    # reverse if needed
## 迭代写法
def dfs(root):
    stack = [node]
    visited = set()
    while stack:
        cur_node = stack.pop()
        visited.add(cur_node)
		process(node)
        nodes = generate_related_nodes(node)
        stack.push(nodes)
# 广度优先算法模板
def bfs(root):
    queue = []
    quene.add([start])
    visited.add(start)
    
    while queue:
        node = queue.pop()
        visited.add(node)
        process(node)
        nodes = generate_related_nodes(node)
        quene.push(nodes)
```

## 4.二分查找

```python
# 二分查找
left, right = 0, len(array)-1
while left <= right:
    mid = (left + right) // 2
    if array[mid] == target:
        return result or break
    elif array[mid] < target:
        left = mid + 1
    else:
        right = mid - 1	
```

## 5.字典树(Trie)的实现

```python
class Trie:
    def __init__(self):
        self.root = {}
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node["#"] = None
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        if "#" not in node:
            return False
        return True
    def startswith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
```

## 6.并查集(disjoint set)

```python
# Python 
def init(p): 
	# for i = 0 .. n: p[i] = i; 
	p = [i for i in range(n)] 
 
def union(self, p, i, j): 
	p1 = self.parent(p, i) 
	p2 = self.parent(p, j) 
	p[p1] = p2 
 
def parent(self, p, i): 
	root = i 
	while p[root] != root: 
		root = p[root] 
	while p[i] != i: # 路径压缩 ?
		x = i
        i = p[i]
        p[x] = root 
	return root
```

## 7.双向DFS

```python
quene_start, quene_end = collections.deque([start]), collections.deque([end])
level_start, level_end = {start}, {end}
visit = {start, end}
count = 1
while len(quene_start) != 0:
    for _ in range(len(quene_start)):
        node = quene_start.pop()
        for new_node in relate[node]:
            if new_node in level_end:
                retrun count
            if new_node not in visit:
                visit.add(new_node)
                quene_start.append(new_node)
    count += 1
    level_start = set(quene_start)
    if len(quene_start) > len(quene_end):
        quene_start, quene_end = quene_end, quene_start
        level_start, level_end = level_end, level_start
```

## 8.布隆过滤器(bloom filter)

```python
from bitarray import bitarray 
import mmh3 
class BloomFilter: 
	def __init__(self, size, hash_num): 
		self.size = size 
		self.hash_num = hash_num 
		self.bit_array = bitarray(size) 
		self.bit_array.setall(0) 
	def add(self, s): 
		for seed in range(self.hash_num): 
			result = mmh3.hash(s, seed) % self.size 
			self.bit_array[result] = 1 
	def lookup(self, s): 
		for seed in range(self.hash_num): 
			result = mmh3.hash(s, seed) % self.size 
			if self.bit_array[result] == 0: 
				return "Nope" 
		return "Probably" 
bf = BloomFilter(500000, 7) 
bf.add("dantezhao") 
print (bf.lookup("dantezhao")) 
print (bf.lookup("yyj")) 
```

## 9.LRU Cache(LRU-least recently used)

```python
class LRUCache(object): 

	def __init__(self, capacity): 
		self.dic = collections.OrderedDict() 
		self.remain = capacity

	def get(self, key): 
		if key not in self.dic: 
			return -1 
		v = self.dic.pop(key) 
		self.dic[key] = v   # key as the newest one 
		return v 

	def put(self, key, value): 
		if key in self.dic: 
			self.dic.pop(key) 
		else: 
			if self.remain > 0: 
				self.remain -= 1 
			else:   # self.dic is full
				self.dic.popitem(last=False) 
		self.dic[key] = value
```

## 10.排序算法

```python
## 冒泡排序
def bubble_sort(nums):
    for i in range(len(nums) - 1):
        for j in range(len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
## 选择排序
def selection_sort(nums):
    for i in range(len(nums) - 1):
        min_idx = i
        for j in range(i + 1, len(nums)):
            if nums[min_idx] > nums[j]:
                min_idx = j
        nums[i], nums[min_idx] = nums[min_idx], nums[i]
## 插入排序
def insert_sort(nums):
    for i in range(len(nums) - 1):
        pre, cur = i, i + 1
        while nums[pre] > nums[cur] and pre >= 0:
            nums[pre + 1] = nums[pre]
            pre -= 1
        nums[pre] = nums[cur]        
## 快速排序
def quick_sort(begin, end, nums):
    if begin >= end:
        return
    pivot_index = partition(begin, end, nums)
    quick_sort(begin, pivot_index-1, nums)
    quick_sort(pivot_index+1, end, nums)
    
def partition(begin, end, nums):
    pivot = nums[begin]
    mark = begin
    for i in range(begin+1, end+1):
        if nums[i] < pivot:
            mark +=1
            nums[mark], nums[i] = nums[i], nums[mark]
    nums[begin], nums[mark] = nums[mark], nums[begin]
    return mark
## 归并排序
def mergesort(nums, left, right):
    if right <= left:
        return
    mid = (left+right) >> 1
    mergesort(nums, left, mid)
    mergesort(nums, mid+1, right)
    merge(nums, left, mid, right)

def merge(nums, left, mid, right):
    temp = []
    i = left
    j = mid+1
    while i <= mid and j <= right:
        if nums[i] <= nums[j]:
            temp.append(nums[i])
            i +=1
        else:
            temp.append(nums[j])
            j +=1
    while i<=mid:
        temp.append(nums[i])
        i +=1
    while j<=right:
        temp.append(nums[j])
        j +=1
    nums[left:right+1] = temp
## 堆排序
def heapify(parent_index, length, nums):
    temp = nums[parent_index]
    child_index = 2*parent_index+1
    while child_index < length:
        if child_index+1 < length and nums[child_index+1] > nums[child_index]:
            child_index = child_index+1
        if temp > nums[child_index]:
            break
        nums[parent_index] = nums[child_index]
        parent_index = child_index
        child_index = 2*parent_index + 1
    nums[parent_index] = temp


def heapsort(nums):
    for i in range((len(nums)-2)//2, -1, -1):
        heapify(i, len(nums), nums)
    for j in range(len(nums)-1, 0, -1):
        nums[j], nums[0] = nums[0], nums[j]
        heapify(0, j, nums)
```

# 三、记忆理解知识

## 1.图的属性和分类

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

## 2.启发式搜索

1. 估价函数：评价哪些结点是我们最希望找到的结点，返回一个非负实数，可认为是结果n到目标路径的成本
2. 估价函数是告知智能搜索方向的方法，提供一种明智的方法来猜测哪个邻居结点会导向一个目标

## 3.AVL树

1. 发明者 G.M.Adelson-Velsky和Evgenii Landis
2. 平衡因子，左子树高度减右子树高度（或相反）
3. 通过旋转操作来进行平衡（四种）

- 左旋： 右右子树——>左旋
- 右旋： 左左子树——>右旋
- 左右旋： 左右子树——>左旋+右旋
- 右左旋： 右左子树——>右旋+左旋

4. 不足：存储额外信息，且调整次数频繁——>近似平衡二叉树

## 4.红黑树

1. 任何一个结点左右子树的高度差小于2倍
2. 根结点是黑色
3. 每个叶 结点（ NILNILNIL结点，空 结点）是黑色的 
4. 不能有相邻接的两个红色 结点
5. 从任一 结点到其每个叶子的所有路径都包含相同数目黑色 结点

## 5.位运算符

1. 左移 << 高位移除，低位补零
2. 右移 >> 低位移除，高位补零
3. 按位或 | 
4. 按位与 &
5. 按位反 ~
6. 按位异或（相同为零不同为一） ^ 

- 异或也可用**不进位加法**理解，相同为零，不同为一
- 异或操作的一些特点
  - x ^ 0 = x
  - x ^ 1s = ~x    // 注意 1s = ~ 0
  - x ^ (~x) = 1s
  - x ^ x = 0
  - c = a ^ b   a = b ^ c    b = a ^ c
  - a ^ b ^ c = a ^ (b ^ c) = (a ^ b) ^ c

7. 指定位置的位运算

- 将x最右边的n位清零 x & (~0 << n)
- 获取x的第n位值（0或者1）： (x>>n) & 1
- 获得x的第n位的幂值： x & (1<<n)
- 仅将第n位置为1： x | (1<<n)
- 仅将第n位置为0： x &(~(1<<n))
- 将x最高位至第n位（含）清零： x & ((1<<n)-1)

## 6.动态规划

- 分治的问题在过程中可以淘汰次优解，可能使用动态规划求解
- 1. 分解复杂问题到多个简单子问题
  2. 分治 + 最优子结构
  3. 动态递归（由简入繁）

**动态规划和递归、分治没有根本上的区别（关键看有无最优的子结构）**

**拥有共性：找出重复子问题**

**差异性：最优子结构，中途可以淘汰次优解**

**多种情况的动态规划的状态转移方程串讲**

1. 爬楼梯——》一维DP
2. 不同路径——》二维DP
3. 打家劫舍——》一维DP or 二维DP(增加一个维度提供操作信息)
4. 最小路径和——》二维DP（选择）
5. 股票买卖——》多维DP

**进阶版的动态规划习题**

**复杂度的来源**

1. 状态拥有更多维度（二维、三维或者更多、甚至需要压缩）
2. 状态方差更加复杂

**本质：内功、逻辑思维、数学**

## 7.字符串算法

字符串基础知识和引申题目

**Python Java 字符串immutable不可变，C++可变，不可变则修改字符串后形成新的字符**

1. 字符串操作问题
2. 异位词问题
3. 回文串问题

**高级字符串算法**

1. 最长子串、子序列问题
2. 字符串+DP问题

**字符串匹配算法**

1. 暴力匹配法：O(m*n)
2. Rabin-Karp：子串哈希对比，若相同再暴力比较
3. KMP算法：最长公共前缀和后缀，建立位移表，比较进行位移
4. Boyer-Moore算法：从后比较，并分析最后