# 7-13 字典树和并查集

## Trie 字典树

### 字典树的数据结构

1. 树的搜素： DFS/BFS
2. 二叉搜索树：所有左子树节点小于根，所有右子树节点大于根，中序遍历恰好升序，特点搜索快（O(logN)）
3. 字典树应用场景之一：词的前缀推词
4. 字典树，又称Trie树，又称单词查找树或键树，是一种树形结构。典型应用是用于统计和排序大量的字符串（但不仅限于字符串），所以经常被搜索引擎系统用于文本词频统计。**优点：最大限度地减少无谓的字符串比较，查询效率比哈希表高。**Trie不是二叉树，是多叉树

### 字典树的核心思想

1. Trie的核心思想：空间换时间
2. 利用字符串的公共前缀来降低查询时间的开销以达到提高效率的目的

### 字典树的基本性质

1. 结点本身不存完整单词；
2. 从根结点到某一节点上，路径上经过的字符连接起来，为该结点对应的字符串；
3. 每个结点所有子结点路径代表的字符都不相同。
4. 结点可统计额外信息，例如词频

### Python Trie实现

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

### 单词搜索2

```python
## 时间复杂度为 O(m*n*4*3^(k-1))，其中m,n分别为二维网格行数和列数，k为单词平均长度
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = {}
        for s in words:
            node = root
            for c in s:
                if c not in node:
                    node[c] = {}
                node = node[c]
            node["#"] = None

        def dfs(i, j, s, parent, board, res):
            if board[i][j] != "@" and board[i][j] in parent:
                letter = board[i][j]
                s += letter
                node = parent[letter]
                board[i][j] = "@"                
                if "#" in node:
                    res.append(s)
                    node.pop("#")
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x, y = i + dx, j + dy
                    if 0 <= x < m and 0 <= y < n:
                        dfs(x, y, s, node, board, res)
                board[i][j] = letter
                if len(node) == 0:
                    parent.pop(letter)

        m, n = len(board), len(board[0])
        res = []
        for i in range(m):
            for j in range(n):
                dfs(i, j, "", root, board, res)
        return res
```



## 并查集（disjoint set）

并查集，熟记

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



### 适用场景

1. 组团
2. 配对
3. 微信好友、朋友圈、是否是好友？

### 基本操作

1. makeset()新建一个集合，其中包含s个单元素集合
2. union(x,y)合并，将指定的2个元素的集合合并，2个集合不相交，相交则不合并
3. find(x)找到x所在的集合，可用于判断2个元素是否处于同一集合（通过比较2元素的代表是否相同）

初始化——>每一个元素有一个parent的数组，指向自己

合并——>一直找parent，直到parent==自己，找到代表，将两个集合其中一个代表的parent指向两个一个代表

路径压缩——>路径上所有元素指向代表

# 7-14 高级搜索

### 初级搜索

1. 朴素搜索
2. 优化方式：不重复（fibonacci），剪枝（生成括号问题）
3. 搜索方向：
   - DFS
   - BFS

双向搜素 启发式搜索 （优先级搜索、A*算法）



## 剪枝

回溯法采用试错的思想，它尝试分步的去解决一个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效的正确的解答的时候，它将取消上一步甚至是上几步的计算，再通过其它的可能的分步解答再次尝试寻找问题的答案。
回溯法通常用最简单的递归方法来实现，在反复重复上述的步骤后可能出现两种情况：
• 找到一个可能存在的正确的答案
• 在尝试了所有可能的分步方法后宣告该问题没有答案在最坏的情况下，回溯法会导致一次复杂度为指数时间的计算。

## 双向DFS

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



## 启发式搜索

1. 估价函数：评价哪些结点是我们最希望找到的结点，返回一个非负实数，可认为是结果n到目标路径的成本
2. 估价函数是告知智能搜索方向的方法，提供一种明智的方法来猜测哪个邻居结点会导向一个目标

# 7-15 红黑树和AVL树

## AVL树

1. 发明者 G.M.Adelson-Velsky和Evgenii Landis
2. 平衡因子，左子树高度减右子树高度（或相反）
3. 通过旋转操作来进行平衡（四种）

- 左旋： 右右子树——>左旋
- 右旋： 左左子树——>右旋
- 左右旋： 左右子树——>左旋+右旋
- 右左旋： 右左子树——>右旋+左旋

4. 不足：存储额外信息，且调整次数频繁——>近似平衡二叉树

## 红黑树

1. 任何一个结点左右子树的高度差小于2倍
2. 根结点是黑色
3. 每个叶 结点（ NILNILNIL结点，空 结点）是黑色的 
4. 不能有相邻接的两个红色 结点
5. 从任一 结点到其每个叶子的所有路径都包含相同数目黑色 结点



