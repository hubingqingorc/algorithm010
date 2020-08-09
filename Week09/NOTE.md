# 9-19 高级动态规划

## 动态规划的复习；附带递归、分治

- 分治的问题在过程中可以淘汰次优解，可能使用动态规划求解
- 1. 分解复杂问题到多个简单子问题
  2. 分治 + 最优子结构
  3. 动态递归（由简入繁）

**动态规划和递归、分治没有根本上的区别（关键看有无最优的子结构）**

**拥有共性：找出重复子问题**

**差异性：最优子结构，中途可以淘汰次优解**

## 多种情况的动态规划的状态转移方程串讲

1. 爬楼梯——》一维DP
2. 不同路径——》二维DP
3. 打家劫舍——》一维DP or 二维DP(增加一个维度提供操作信息)
4. 最小路径和——》二维DP（选择）
5. 股票买卖——》多维DP

## 进阶版的动态规划习题

### 复杂度的来源

1. 状态拥有更多维度（二维、三维或者更多、甚至需要压缩）
2. 状态方差更加复杂

**本质：内功、逻辑思维、数学**

# 9-20 字符串算法

## 字符串基础知识和引申题目

**Python Java 字符串immutable不可变，C++可变，不可变则修改字符串后形成新的字符**

1. 字符串操作问题
2. 异位词问题
3. 回文串问题

## 高级字符串算法

1. 最长子串、子序列问题
2. 字符串+DP问题

## 字符串匹配算法

1. 暴力匹配法：O(m*n)
2. Rabin-Karp：子串哈希对比，若相同再暴力比较
3. KMP算法：最长公共前缀和后缀，建立位移表，比较进行位移
4. Boyer-Moore算法：从后比较，并分析最后



**不同路径-二 状态转移方程**

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        if m == 0:
            return 0
        n = len(obstacleGrid[0])
        if n == 0:
            return 0
        if obstacleGrid[0][0] == 1 or obstacleGrid[-1][-1] == 1:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][1] = 1
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 0:
                    dp[i+1][j+1] = dp[i][j+1] + dp[i+1][j]
        return dp[-1][-1]
```

