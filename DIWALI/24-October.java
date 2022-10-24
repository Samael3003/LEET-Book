

## 1710. Maximum Units on a Truck

    class Solution {
        public int maximumUnits(int[][] boxTypes, int truckSize) {
         Arrays.sort(boxTypes, (a, b) -> b[1] - a[1]);
          int ans = 0;
          for(int[] b : boxTypes) {
            int count = Math.min(b[0], truckSize);
            ans += count * b[1];
            truckSize -= count;
            if(truckSize == 0) return ans;
          }
          return ans;
        }
    }
  

## 1465. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts

    class Solution {
        public int maxArea(int h, int w, int[] horizontalCuts, int[] verticalCuts) 
        {
            Arrays.sort(horizontalCuts);

            Arrays.sort(verticalCuts);


            int max = horizontalCuts[0];

            for(int i=1;i<horizontalCuts.length;i++)

                max = Math.max(horizontalCuts[i]-horizontalCuts[i-1],max);

            max = Math.max(h-horizontalCuts[horizontalCuts.length-1],max);


            int maxv = verticalCuts[0];
            for(int i=1;i<verticalCuts.length;i++)
                maxv = Math.max(verticalCuts[i]-verticalCuts[i-1],maxv);
            maxv = Math.max(w-verticalCuts[verticalCuts.length-1],maxv);


            return (int)(1L*max*maxv %1000000007);
        }
    }


376. Wiggle Subsequence

    class Solution {
        public int wiggleMaxLength(int[] nums) {
            if(nums.length<2)return 1;

            int count=1;

            int prevDiff=0;
            for(int i=1;i<nums.length;i++){
                int diff = nums[i] - nums[i-1];
                if((diff>0 && prevDiff <= 0) || (diff<0 && prevDiff >= 0)){
          // the equals in prevDiff <= 0 && prevDiff >=0 can only be used at the first iteration, otherwise prevDiff will never be zero afterwards
                    count++;
                    prevDiff = diff;
                }
            }
            return count;
        }
    }


## 135. Candy

    class Solution {
        public int candy(int[] ratings) {
            int n =  ratings.length;
            if(n==0)
                return 0;
            int left[] = new int[n];
            Arrays.fill(left,1);
            for(int i=1;i<n;i++)
            {
                if(ratings[i] > ratings[i-1])
                    left[i] = left[i-1]+1;
            }
            int right[] = new int[n];
            Arrays.fill(right,1);
            for(int i=n-2;i>=0;i--)
            {
                if(ratings[i] > ratings[i+1])
                    right[i] = right[i+1]+1;
            }
            int ans = 0;
            for(int i=0;i<n;i++)
                ans += Math.max(left[i],right[i]);
            return ans;
        }
    }



## 128. Longest Consecutive Sequence

    class Solution {
        public int longestConsecutive(int[] nums) {
          Set <Integer> set = new HashSet <Integer>();
          for(int n : nums)
            set.add(n);

          int streak = 0;
          for(int n : nums) {
            if(!set.contains(n - 1)) {
              int currNum = n;
              int currStreak = 1;

              while(set.contains(currNum + 1)) {
                currNum ++;
                currStreak ++;
              }

              streak = Math.max(streak , currStreak);
          }
        }
          return streak;
      }
    }
