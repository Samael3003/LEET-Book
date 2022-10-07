# DAY - 1

### 136. Single Number
        class Solution(object):
            def singleNumber(self, nums):
                res = 0
                for i in nums:
                    res ^= i
                return res

### 169. Majority Element
        class Solution(object):
            def majorityElement(self, nums):
                dup = list(dict.fromkeys(nums))
                for i in dup:
                    tmp = nums.count(i)
                    if tmp > len(nums)/2:
                        return i


### 15. 3Sum
        class Solution(object):
            def threeSum(self, nums):
                """
                :type nums: List[int]
                :rtype: List[List[int]]
                """
                # special case
                if len(nums) <= 2:
                    return []

                result = set()  # result will contain the unique tuples

                nums = sorted(nums) # nlog(n) complexity

                for x in range(len(nums)-2):
                    b = x+1          # begining pointer points to next
                    e = len(nums)-1  # end pointer points to last element in nums

                    while b < e: # while pointers don't cross one another
                        val = nums[x]+ nums[b] + nums[e]
                        if val == 0: # we have found the sum
                            result.add(tuple(sorted((nums[x],nums[b],nums[e])))) # need to convert to tuple as it's immutable

                            b += 1
                            e -= 1

                        if val < 0: # we need to increase the result, so move towards the right side
                            b += 1
                        if val > 0:  # we need to decrease the result, so move towards the left side
                            e -= 1
                return result

# DAY - 2

### 75. Sort Colors
        class Solution:
            def sortColors(self, nums: List[int]) -> None:
                """
                Do not return anything, modify nums in-place instead.
                """
                l, r = 0, len(nums) - 1
                i = 0
                while i <= r:
                    if nums[i] == 0:
                        nums[i], nums[l] = nums[l], nums[i]
                        l += 1
                        i += 1
                    elif nums[i] == 1:
                        i += 1
                    else:
                        nums[i], nums[r] = nums[r], nums[i]
                        r -= 1
                    # Time: O(n)
                    # Space: O(1)


### 56. Merge Intervals
        class Solution:
            def merge(self, intervals: List[List[int]]) -> List[List[int]]:
                intervals.sort()
                res = [intervals[0]]
                for i in range(1, len(intervals)):
                    if intervals[i][0] <= res[-1][1]:
                        res[-1][1] = max(res[-1][1], intervals[i][1])
                    else:
                        res.append(intervals[i])
                return res
            # Time: O(n * log(n)) where n is the length of intervals
            # Space: O(n) it's the space taken by res


### 706. Design HashMap
        class MyHashMap:
            map: dict[int, int]

            def __init__(self):
                self.map = {}

            def put(self, key: int, value: int) -> None:
                if key not in self.map:
                    self.map.setdefault(key, value)
                else:
                    self.map[key] = value

            def get(self, key: int) -> int:
                if key in self.map:
                    return self.map[key]
                else:
                    return -1

            def remove(self, key: int) -> None:
                if key in self.map:
                    self.map.pop(key)

        # Your MyHashMap object will be instantiated and called as such:
        # obj = MyHashMap()
        # obj.put(key,value)
        # param_2 = obj.get(key)
        # obj.remove(key)

