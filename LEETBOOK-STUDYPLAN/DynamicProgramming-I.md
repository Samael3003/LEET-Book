# DAY - 1


### 509. Fibonacci Number
        class Solution(object):
            def fib(self, n):
                """
                :type n: int
                :rtype: int
                """
                if n <= 1: return n
                a, b = 0, 1
                res = 0
                for _ in range(2, n + 1):
                    res = a + b
                    a, b = b, res
                return res
        
        
### 1137. N-th Tribonacci Number    
        class Solution(object):
            def tribonacci(self, n):
                """
                :type n: int
                :rtype: int
                """
                if n == 0:
                    return 0
                res = []

                for i in range(n+1):
                    if i == 0:
                        res.append(0)
                    elif i <= 2:
                        res.append(1)
                    else:
                        next_number = res[-1] + res[-2] + res[-3]
                        res.append(next_number)

                return res[n]  
                
# DAY - 2

### 70. Climbing Stairs
                class Solution:
                    def climbStairs(self, n: int) -> int:
                        if(n<=2):
                            return n
                        l=1
                        r=2
                        curr=0
                        for i in range(2,n):
                            curr=l+r
                            l=r
                            r=curr
                        return curr
        
        
### 746. Min Cost Climbing Stairs
                class Solution:
                    def minCostClimbingStairs(self, cost: List[int]) -> int:

                        dp = [0]*len(cost)        
                        dp [-1], dp [-2] = cost[-1], cost[-2]

                        for i in range(len(cost)-3, -1, -1):
                            dp[i] = min(dp[i+1], dp[i+2]) + cost[i]
                        return min(dp[0],dp[1])

# DAY - 3

### 198. House Robber
                class Solution:
                    def rob(self, nums: List[int]) -> int:
                        if len(nums) == 1:
                            return nums[0]
                        first, second = nums[0], max(nums[0], nums[1])
                        res = second
                        for i in range(2, len(nums)):
                            first, second = second, max(nums[i] + first, second)
                            res = max(res, second)
                        return res
                    # Time: O(n)
                    # Space: O(1)


### 213. House Robber II
                class Solution:
                    def rob(self, nums: List[int]) -> int:
                        # Check if only one element
                        if len(nums) == 1:
                            return nums[0]

                        # First iteration
                        rob1, rob2 = 0, 0
                        for i in nums[:-1]:
                            temp = max(i+rob1, rob2)
                            rob1 = rob2
                            rob2 = temp
                        res1 = rob2
                        # Second iteration
                        rob1, rob2 = 0, 0
                        for i in nums[1:]:
                            temp = max(i+rob1, rob2)
                            rob1 = rob2
                            rob2 = temp
                        res2 = rob2
                        return max(res1,res2)



### 740. Delete and Earn
                # just like the bank robber question  here we cannot take two consecutive numbers  
                from collections import defaultdict
                def fun(dc,i,n,memo):
                    if(i>n):
                        return 0
                    if(i in memo):
                        return(memo[i])
                    a=fun(dc,i+1,n,memo)
                    b=fun(dc,i+2,n,memo)+dc[i]
                    memo[i]=max(a,b)
                    return(max(a,b))
                class Solution:
                    def deleteAndEarn(self, nums: List[int]) -> int:
                        dc=defaultdict(lambda:0)
                        n=max(nums)
                        for a in nums:
                            dc[a]+=a
                        return(fun(dc,0,n,{}))        


# DAY - 4

### 55. Jump Game
        class Solution(object):
            def canJump(self, nums):
                capacity = nums[0]

                for x in nums[1:]:
                    if capacity <= 0:
                        return False
                    capacity = max(capacity - 1, x)

                return True


### 45. Jump Game II
        class Solution:
            def jump(self, nums: List[int]) -> int:
                if len(nums)==1:
                    return 0
                l,r=1,nums[0]+1
                maxx=r-1
                count=1
                while maxx<len(nums)-1:
                    for i in range(l,r):
                        maxx=max(maxx,i+nums[i])
                    l=r
                    r=maxx+1
                    count+=1
                return count
