# 1.https://leetcode-cn.com/problems/longest-valid-parentheses/
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        ans = 0
        dp = [0] * (len(s))
        for i in range(1, len(s)):
            right = i
            left = i - dp[i-1] - 1
            if s[right] == ')' and left >= 0 and s[left] == '(':
                dp[i] = dp[i-1] + 2 + dp[left-1]
                ans = max(ans, dp[i])
        return ans
# 2.https://leetcode-cn.com/problems/minimum-path-sum/
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = grid
        for i in range(n-2, -1, -1):
            dp[m-1][i] = dp[m-1][i] + dp[m-1][i+1]
        for i in range(m-2, -1, -1):
            dp[i][n-1] = dp[i][n-1] + dp[i+1][n-1]
        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                dp[i][j] = dp[i][j] + min(dp[i+1][j], dp[i][j+1])
        return dp[0][0]
# 3.https://leetcode-cn.com/problems/edit-distance/
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i
        for i in range(1, n + 1):
            dp[0][i] = i
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
        return dp[-1][-1]
# 4.https://leetcode-cn.com/problems/decode-ways
class Solution:
    def numDecodings(self, s: str) -> int:
        length = len(s)
        if s[0] == "0":
            return 0
        dp = [1 for _ in range(len(s)+1)]
        for i in range(1, len(s)):
            if s[i:i+1] == "0":
                if 0 < int(s[i-1:i+1]) <= 26:
                    dp[i+1] = dp[i-1]
                else:
                    return 0
            else:
                if s[i-1:i] == "0":
                    dp[i+1] = dp[i]
                elif 0 < int(s[i-1:i+1]) <= 26:
                    dp[i+1] = dp[i] + dp[i-1]
                else:
                    dp[i+1] = dp[i]
        return dp[-1]
# 5.https://leetcode-cn.com/problems/maximal-square/
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        if m == 0:
            return 0
        else:
            n = len(matrix[0])
            if n == 0:
                return 0
        ans = 0
        for i in range(m):
            if matrix[i][0] == "1":
                ans = 1
                break
        if ans == 0:
            for i in range(n):
                if matrix[0][i] == "1":
                    ans = 1
                    break
        dp = matrix.copy()
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == "1":
                    dp[i][j] = min(int(matrix[i-1][j-1]), int(matrix[i-1][j]), int(matrix[i][j-1])) + 1
                    ans = max(ans, dp[i][j])
        return ans*ans
# 6.https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k/
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])
        ans = -math.inf
        for l in range(n):
            row_sum = [0] * m
            for r in range(l, n):
                for i in range(m):
                    row_sum[i] += matrix[i][r]
                temp, temp_max = row_sum[0], row_sum[0]
                for j in range(1, m):
                    if temp >= 0:
                        temp += row_sum[j]
                    else:
                        temp = row_sum[j]
                    temp_max = max(temp_max, temp)
                if temp_max < k:
                    ans = max(ans, temp_max)
                elif temp_max == k:
                    return k
                else:
                    for p in range(m):
                        temp_sum = 0
                        for q in range(p, m):
                            temp_sum += row_sum[q]
                            if temp_sum < k:
                                ans = max(ans, temp_sum)
                            elif temp_sum == k:
                                return k

        return ans
# 7.https://leetcode-cn.com/problems/frog-jump/
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        dp = collections.defaultdict(set)
        for i in stones:
            dp[i] = set()
        dp[0].add(0)
        for i in stones:
            temp = list(dp[i])
            for j in range(len(temp)):
                for step in range(temp[j]-1, temp[j]+2):
                    pos = i + step
                    if pos in dp:
                        dp[pos].add(step)
        return len(dp[stones[-1]]) != 0
# 8.https://leetcode-cn.com/problems/split-array-largest-sum
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        length = len(nums)
        nums_sum = [0] * (length + 1)
        for i in range(length):
            nums_sum[i + 1] = nums_sum[i] + nums[i]
        dp = [[sys.maxsize] * (m + 1) for _ in range(length + 1)]
        dp[0][0] = 0
        for i in range(1, length + 1):
            for j in range(1, m + 1):
                for k in range(i):
                    dp[i][j] = min(dp[i][j], max(dp[k][j-1], nums_sum[i] - nums_sum[k]))
        return dp[length][m]
# 9.https://leetcode-cn.com/problems/student-attendance-record-ii/
class Solution:
    def checkRecord(self, n: int) -> int:
        m = 1000000007
        dp = [0] * max(4, n + 1)
        dp[0], dp[1], dp[2], dp[3] = 1, 2, 4, 7
        for i in range(4, n + 1):
            dp[i] = (2 * dp[i - 1]) % m + (m - dp[i - 4]) % m
        ans = dp[n]
        for j in range(1, n + 1):
            ans += (dp[j - 1] * dp[n - j]) % m
        return ans % m
# 10.https://leetcode-cn.com/problems/task-scheduler/
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        count = dict()
        for c in tasks:
            if c not in count:
                count[c] = 1
            else:
                count[c] += 1
        max_count, max_value = dict(), 0
        for i in count.values():
            max_value = max(max_value, i)
            if i not in max_count:
                max_count[i] = 1
            else:
                max_count[i] += 1
        num1 = (max_value - 1) * (n + 1) + max_count[max_value]
        num2 = len(tasks)
        return max(num1, num2)
# 11.https://leetcode-cn.com/problems/palindromic-substrings/
class Solution:
    def countSubstrings(self, s: str) -> int:
        length = len(s)
        if length == 0:
            return 0
        dp = [[False] * length for _ in range(length)]
        for i in range(length):
            dp[i][i] = True
        ans = length
        for i in range(length-2, -1, -1):
            for j in range(i+1, length):
                if s[i] == s[j]:
                    if j - i == 1:
                        dp[i][j] = True
                        ans += 1
                    else:
                        dp[i][j] = dp[i+1][j-1]
                        if dp[i][j] == True:
                            ans += 1
                else:
                    dp[i][j] = False
        return ans
# 13.https://leetcode-cn.com/problems/burst-balloons/
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        rec = [[0] * (n + 2) for _ in range(n + 2)]
        val = [1] + nums + [1]
        for i in range(n - 1, -1, -1):
            for j in range(i + 2, n + 2):
                for k in range(i + 1, j):
                    total = val[i] * val[k] * val[j]
                    total += rec[i][k] + rec[k][j]
                    rec[i][j] = max(rec[i][j], total)

        return rec[0][n + 1]
