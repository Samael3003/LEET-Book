

## 12. Integer to Roman

    class Solution {
        private static final StringBuilder sb = new StringBuilder(15);

        private static final String[] thou = {"", "M", "MM", "MMM"};
        private static final String[] hund = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC","DCCC","CM"};
        private static final String[] tens = {"", "X", "XX", "XXX", "XL", "L", "LX","LXX","LXXX","XC"};
        private static final String[] ones = {"", "I", "II", "III", "IV", "V", "VI","VII","VIII","IX"};

        public String intToRoman(int num) {
            sb.setLength(0);

            sb.append(thou[num/1000]);
            sb.append(hund[(num%1000)/100]);
            sb.append(tens[(num%100)/10]);
            sb.append(ones[num%10]);

            return sb.toString();
        }
    }


## 237. Delete Node in a Linked List

    class Solution {
        public void deleteNode(ListNode node) {
            node.val=node.next.val;
            node.next=node.next.next;
        }
    }


## 2095. Delete the Middle Node of a Linked List

    class Solution {
        public ListNode deleteMiddle(ListNode head) {

            //Code here....
            if(head == null || head.next == null)
                return null;    //Checking corner case

            ListNode curr = head;
            curr = middle(curr);    //Finding the element previous to the middle node
            curr.next = curr.next.next; //Removing the middle node

            return head;

        }

        public ListNode middle(ListNode head)
        {
            ListNode slow = head, fast = head,prev = null;
            while(fast != null && fast.next != null)
            {
                prev = slow;
                slow = slow.next;
                fast = fast.next.next;
            }

            return prev;
        }
    }


## 1531. String Compression II

    class Solution {
        private int[][] dp;
        private char[] chars;
        private int n;

        public int getLengthOfOptimalCompression(String s, int k) {
            this.chars = s.toCharArray();
            this.n = s.length();
            this.dp = new int[n][k+1];
            for (int[] row: dp) {
                Arrays.fill(row, -1);
            }
            return dp(0, k);
        }

        private int dp(int i, int k) {
            if (k < 0) return n;
            if (n <= i + k) return 0;

            int ans = dp[i][k];
            if (ans != -1) return ans; 
            ans = dp(i + 1, k - 1);
            int length = 0, same = 0, diff = 0;

            for (int j=i; j < n && diff <= k; j++) {

                if (chars[j] == chars[i]) {
                    same++;
                    if (same <= 2 || same == 10 || same == 100) length++;
                } else {
                    diff++; 
                }
                ans = Math.min(ans, length + dp(j + 1, k - diff)); 
            }
            dp[i][k] = ans;
            return ans;
        }
    }


## 1335. Minimum Difficulty of a Job Schedule

    class Solution {
        public int minDifficulty(int[] jobDifficulty, int d) {
            int len = jobDifficulty.length;
            if (d > len) return -1;
            int[][] minDifficulty = new int[d][len];
            for (int i = 1; i < d; i++) {
                Arrays.fill(minDifficulty[i], Integer.MAX_VALUE);
            }
            int maxDifficulty = 0;
            for (int i = 0; i <= len - d; i++) {
                maxDifficulty = Math.max(maxDifficulty, jobDifficulty[i]);
                minDifficulty[0][i] = maxDifficulty;
            }
            for (int day = 1; day < d; day++) {
                for (int to = day; to <= len - d + day; to++) {
                    int currentDayDifficulty = jobDifficulty[to];
                    int result = Integer.MAX_VALUE;
                    for (int j = to - 1; j >= day - 1; j--) {
                        result = Math.min(result, minDifficulty[day - 1][j] + currentDayDifficulty);
                        currentDayDifficulty = Math.max(currentDayDifficulty, jobDifficulty[j]);
                    }
                    minDifficulty[day][to] = result;
                }   
            }
            return minDifficulty[d - 1][len - 1];
        }
    }



## 1832. Check if the Sentence Is Pangram
    class Solution {
        public boolean checkIfPangram(String sentence) {
            if(sentence.length()<26) return false;
             HashSet<Character> set = new HashSet<>();
            for(char c:sentence.toCharArray()){
                set.add(c);
            }

            return set.size()==26;

        }
    }


## 38. Count and Say

    class Solution {
        public String countAndSay(int n) {
            String s = "1";
            for(int i=2;i<=n;i++){
                s = countIndex(s);
            }
            return s;
        }

        public String countIndex(String s){
            StringBuilder sb = new StringBuilder();
            char c = s.charAt(0);
            int count = 1;
            for(int i=1;i<s.length();i++){
                if(s.charAt(i) == c)
                    count++;
                else{
                    sb.append(count);
                    sb.append(c);
                    c = s.charAt(i);
                    count=1;
                }
            }
            sb.append(count);
            sb.append(c);
            return sb.toString();
        }
    }



## 692. Top K Frequent Words

    class Solution {
        public List<String> topKFrequent(String[] words, int k) {
                TreeMap<String, Integer> map = new TreeMap<>(String::compareTo);
                Arrays.stream(words).forEach(x -> map.put(x, map.getOrDefault(x, 0) + 1));
                return map.entrySet().stream()
                    .sorted((o1, o2) -> Integer.compare(o2.getValue(), o1.getValue()))
                    .map(Map.Entry::getKey)
                    .limit(k)
                    .collect(Collectors.toList());
            }
    }



## 12. Integer to Roman

    class Solution {
        private static final StringBuilder sb = new StringBuilder(15);

        private static final String[] thou = {"", "M", "MM", "MMM"};
        private static final String[] hund = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC","DCCC","CM"};
        private static final String[] tens = {"", "X", "XX", "XXX", "XL", "L", "LX","LXX","LXXX","XC"};
        private static final String[] ones = {"", "I", "II", "III", "IV", "V", "VI","VII","VIII","IX"};

        public String intToRoman(int num) {
            sb.setLength(0);

            sb.append(thou[num/1000]);
            sb.append(hund[(num%1000)/100]);
            sb.append(tens[(num%100)/10]);
            sb.append(ones[num%10]);

            return sb.toString();
        }
    }  





## 2289. Steps to Make Array Non-decreasing

    class Solution {
        public int totalSteps(int[] nums) {
            int max=0;
            Stack<Integer> st = new Stack<>();
            int[] dp = new int[nums.length];
            st.push(nums.length-1);
            for(int i=nums.length-2;i>=0;i--){
                if(nums[st.peek()]>=nums[i]){
                    st.push(i);
                }else{
                    while(!st.isEmpty() && nums[st.peek()]<nums[i]){
                        dp[i] = Math.max(dp[i]+1,dp[st.peek()]);
                        st.pop();
                    }
                    max = Math.max(max,dp[i]);
                    st.push(i);
                }
            }
            return max;
        }
    }





## 2280. Minimum Lines to Represent a Line Chart

    class Solution {
        public int maximumTop(int[] nums, int k) {
            if(k % 2 == 1 && nums.length == 1) return -1;
            int ans = -1;
            if(k > nums.length){
                for(int num : nums) ans = Math.max(ans, num);
            }
            else{
                for(int i = 0; i < k - 1; i++){
                    ans = Math.max(ans, nums[i]);
                }
                if(k < nums.length) ans = Math.max(ans, nums[k]);
            }
            return ans;
        }
    }




## 2280. Minimum Lines to Represent a Line Chart

    class Solution {
        // check for collinearity of 3 points is there slope is equal
        // => (y2 - y1) / (x2 - x1) = (y1 - y0) / (x1 - x0)
    //     => (y2 - y1) * (x1 - x0) = (y1 - y0) * (x2 - x1) (cross multiplication)
        // if lhs == rhs then points are collinear else points are not collinear
        public int minimumLines(int[][] stockPrices) {
            if(stockPrices.length == 1) return 0;

            Arrays.sort(stockPrices, (a, b) -> a[0] - b[0]);
            int lines = 1;


            for(int i = 2; i < stockPrices.length; i++){
                int x0 = stockPrices[i][0];
                int x1 = stockPrices[i - 1][0];
                int x2 = stockPrices[i - 2][0];
                int y0 = stockPrices[i][1];
                int y1 = stockPrices[i - 1][1];
                int y2 = stockPrices[i - 2][1];

                int lhs = (y2 - y1) * (x1 - x0);
                int rhs = (y1 - y0) * (x2 - x1);
                if(lhs != rhs){ // points are not collinear  -> they will not lie in same line
                    lines++;
                }
            }
            return lines;
        }
    }




## 1191. K-Concatenation Maximum Sum

    class Solution {
        public int kConcatenationMaxSum(int[] arr, int k) {
            int n = arr.length;
            if(k==1){
                long ans=0L, sum=0L;
                for(int i=0;i<n;i++){
                    sum+=arr[i];
                    ans=Math.max(ans,sum);
                    if(sum<0) sum=0;
                }
                return (int)(ans%1000000007);
            }else{
                ArrayList<Integer> ar = new ArrayList<>();
                for(int i=0;i<n;i++) ar.add(arr[i]);
                long s=0L;
                for(int i=0;i<n;i++) s+=ar.get(i);
                if(s<=0){
                    for(int i=0;i<n;i++) ar.add(ar.get(i));
                    long ans=0L, sum=0L;
                    for(int i=0;i<(n*2);i++){
                        sum+=ar.get(i);
                        ans=Math.max(ans,sum);
                        if(sum<0) sum=0;
                    }
                    return (int)(ans%1000000007);
                }else{
                    for(int i=0;i<n;i++) ar.add(ar.get(i));
                    long ans=0L, sum=0L;
                    for(int i=0;i<(n*2);i++){
                        sum+=ar.get(i);
                        ans=Math.max(ans,sum);
                        if(sum<0) sum=0;
                    }
                    for(int i=0;i<k-2;i++) ans+=s;
                    return (int)(ans%1000000007);
                }
            }
        }
    }





## 665. Non-decreasing Array

    class Solution {
        public boolean checkPossibility(int[] nums) {

            return checkInForwardDirection(nums) || checkInBackwardDirection(nums);
        }

        private boolean checkInForwardDirection(int[] nums){
            int issueIndex = -1;
            int num = 0;
            for(int i = 0; i < nums.length - 1; i++){
                //fix the issue, if found
                if(nums[i] > nums[i + 1]){
                    issueIndex = i;
                    num = nums[i + 1];
                    nums[i + 1] = nums[i]; 
                    break;
                }
            }

            //no issue found, it mean nums is already non - decreasing
            if(issueIndex == -1) return true;

            //check once again, it becomes completely non - decreasing or not
            //after fixing the issue
            if(isNonDecreasing(nums)) return true;

            //restore array back to original
            nums[issueIndex + 1] = num;

            return false;
        }

        private boolean checkInBackwardDirection(int[] nums){

            for(int i = nums.length - 1; i > 0; i--){
                //fix the issue, if found
                if(nums[i] < nums[i - 1]){
                    nums[i - 1] = nums[i];
                    break;
                }
            }

            //check once again, it becomes completely non - decreasing or not
            return isNonDecreasing(nums); 
        }

        private boolean isNonDecreasing(int[] nums){
            for(int i = 0; i < nums.length - 1; i++){
                if(nums[i] > nums[i + 1]) return false;
            }
            return true;
        }
    }




