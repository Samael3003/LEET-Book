# Weekly Contest 310

### 2404. Most Frequent Even Element
        class Solution:
            def mostFrequentEven(self, nums: List[int]) -> int:
                counts = {}
                for num in nums:
                    if num%2 == 1: continue
                    counts[num] = counts.get(num,0)+1
                bigcount = -1
                bigcount_num = -1
                for num, count in counts.items():
                    if count > bigcount:
                        bigcount = count
                        bigcount_num = num
                    elif count == bigcount:
                        bigcount_num = min(num,bigcount_num)
                return bigcount_num
        
### 2405. Optimal Partition of String        
        class Solution:
            def partitionString(self, s):
                """
                :type s: str
                :rtype: int
                """        
                res = 0
                seen = set()
                for c in s:
                    if c not in seen:
                        seen.add(c)
                    else:
                        seen.clear()
                        res += 1
                        seen.add(c)
                res += 1
                return res
        
        
### 2406. Divide Intervals Into Minimum Number of Groups
        class Solution(object):
            def minGroups(self, intervals: List[List[int]]) -> int:
                intervals.sort()
                pq = [intervals[0][1]]
                for i in range(1, len(intervals)):
                    if intervals[i][0] > pq[0]:
                        heappop(pq)
                    heapq.heappush(pq, intervals[i][1])
                return len(pq)  
        
        
### 300. Longest Increasing Subsequence
        class Solution:
            def lengthOfLIS(self, nums: List[int]) -> int:
                if not nums:
                    return 0
                dp = [1] * len(nums)
                for i in range(1, len(nums)):
                    for j in range(0, i):
                        if nums[i] > nums[j]:
                            dp[i] = max(dp[i], dp[j] + 1)
                return max(dp)        
