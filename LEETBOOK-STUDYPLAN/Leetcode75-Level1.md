# DAY - 1

### 1480. Running Sum of 1d Array
        class Solution(object):
            def runningSum(self, nums):
                list_1 = []
                list_1.append(nums[0])
                for i in range(1,len(nums)):
                    nums[i] += nums[i-1]
                    list_1.append(nums[i])
                return list_1

### 724. Find Pivot Index
        class Solution:
            def pivotIndex(self, nums: List[int]) -> int:
                pivot=0
                for i, num in enumerate(nums):
                    if sum(nums)==num+2*pivot:
                        return i
                    pivot+=num
                return -1

# DAY - 2

### 205. Isomorphic Strings
        > IN JAVASCRIPT
        
        var isIsomorphic = function (s, t) {
            let char1Count = {};
            let char2Count = {};
            let charObject = {};
            for (let i = 0; i < s.length; i++) {
                if (!(charObject.hasOwnProperty(s[i]))) {
                    charObject[s[i]] = t[i];
                }
                else {
                    if (charObject[s[i]] == t[i]) {
                        continue;
                    }
                    return false;
                }
                if (char1Count.hasOwnProperty(s[i])) {
                    ++char1Count[s[i]];
                }
                else {
                    char1Count[s[i]] = 1;
                }
                if (char2Count.hasOwnProperty(t[i])) {
                    ++char2Count[t[i]];
                }
                else {
                    char2Count[t[i]] = 1;
                }
            }
            if(Object.keys(char1Count).length == Object.keys(char2Count).length)
            {
                return true;
            }
            return false;
        };


### 392. Is Subsequence
        > IN JAVASCRIPT
        
        function isSubsequence(subsequence, text) {
            if(!subsequence) {
                return true;
            }

            let textPointer = 0;
            let subsequencePointer = 0;

            while(textPointer < text.length) {
                if(subsequence[subsequencePointer] === text[textPointer]) {
                    if(++subsequencePointer === subsequence.length) {
                        return true;
                    }
                }

                textPointer++;
            }

            return false;
        }
