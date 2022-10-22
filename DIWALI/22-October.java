

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



## 2195. Append K Integers With Minimal Sum

	class Solution {
	    public long minimalKSum(int[] nums, int k) {
		TreeSet<Integer> set = new TreeSet<Integer>();
		for (int n : nums)                               
		    set.add(n);
		long sum = ((long)(k + 1) * (long)k) / 2;     
		int count = 0;                                
		for (Integer i : set) {
		    if (i > k)                      
			break;
		    else {
			sum -= i;                              
			count++;                              
		    }                                          
		}
		int i = k + 1;                              
		while (count > 0) {
		    if (!set.contains(i)) {
			sum += i;
			count--;
		    }
		    i++;
		}
		return sum;
	    }
	}



## 2333. Minimum Sum of Squared Difference

	class Solution {
	    public long minSumSquareDiff(int[] nums1, int[] nums2, int k1, int k2) {
		int lo=0,hi=100000,k=k1+k2;
		long extra=0,ans=0;
		while(lo<hi){
		    int mid=(lo+hi)>>1;
		    long need=0;
		    for (int i = 0 ; i< nums1.length&&need<=k;i++){
			need += Math.max(0, Math.abs(nums1[i]-nums2[i])-mid);
		    }
		    if (need<=k){
			hi=mid;
			extra=k-need;
		    }else{
			lo=mid+1;
		    }
		}

		for (int i = 0; i< nums1.length&&lo>0;i++){// make sure to check lo (diff) > 0 here.
		    long diff = Math.min(lo, Math.abs(nums1[i]-nums2[i]));
		    if (diff==lo&&extra>0){ 
			--diff;
			--extra;
		    }
		    ans+=diff*diff;
		}

		return ans;
	    }
	}




## 805. Split Array With Same Average

	class Solution 
	{
	    public boolean splitArraySameAverage(int[] nums)
	    {
		if(nums.length<=1)
		    return false;
		int n = nums.length;
		//finding sum of elements of the array :- nums
		int sum = 0;
		for(int i=0 ; i<n ; i++)
		{
		    sum += nums[i];
		}
		// We divide the given array into 2 parts , where n1 = length of part-1 & s1 = sum of elements of part-1
		// s1 = (sum*n1)/n , since the average of both part-1 & part-2 are equal
		// Range of n1 = [ 1 , n-1 ]
		// We iterate over each possible valid value of n1 , and check if the corresponding subseq of sum = s1 & length = n1 is possible or not
		for(int n1=1 ; n1<=(n-1) ; n1++)
		{
		    int numerator = (sum*n1);
		    int denominator = n;
		    int s1 = numerator/denominator;
		    // if the s1 = float value , then we move on to the next possible n1
		    if(numerator%denominator !=0) 
			continue;
		    //if s1 = integer , then we check if there exists a subseq whose sum = s1 , we check this by dp
		    Boolean dp[][] = new Boolean[n+1][s1+1];
		    boolean isTargetPresent = isPresent( s1 ,0,0,n1, nums, nums.length , dp);
		    if(isTargetPresent)
			return true;
		}
		return false;
	    }
		// function that check if there exists a subseq. of length=n1 & sum=s1
	   private boolean isPresent(int target,int count,int sum,int n1, int[] nums, int n , Boolean dp[][])
	   {
	       if(n==0)
		{
		    if(target==0 && count==n1)
			return true;
		    else
			return false;
		}
		if(count==n1)
		{
		    if(target==0)
			return true;
		    else
			return false;
		}
		if(dp[n][target] != null)
		    return dp[n][target];
		if(nums[n-1] > target)
		{
		    boolean skip = isPresent(target,count,sum,n1,nums,n-1,dp);
		    dp[n][target] = skip;
		    return dp[n][target];
		}
		else
		{
		    boolean include = isPresent(target-nums[n-1] , count+1 , sum + nums[n-1] , n1 , nums , n-1 , dp);
		    boolean exclude = isPresent(target , count , sum , n1 , nums , n-1 , dp );
		    dp[n][target] = include || exclude;
		    return dp[n][target];
		}
	\
	   }
	}



## 47. Permutations II

	class Solution {
	    List<List<Integer>> ml;
	    public List<List<Integer>> permuteUnique(int[] nums) {
		//main list to store permutations.
		ml = new ArrayList<>();
		//visited array to mark an element when it is visited.
		boolean[] visited = new boolean[nums.length];
		//Sort the elements.
		Arrays.sort(nums);

		findPermutations(nums, new ArrayList<>(), visited);

		return ml;
	    }

	    public void findPermutations(int[] nums, List<Integer> cl, boolean[] visited) {
		//When size of child list is equal to length of nums array, meaning a permutation is formed, 
			//so add it in main list.
		if (cl.size() == nums.length) {
		    ml.add(new ArrayList<>(cl));
		    return;
		}

		for (int i = 0; i < nums.length; i++) {
		    //If the element is visited or it is a duplicate, then continue.
		    if (visited[i] || (i > 0 && nums[i] == nums[i - 1]) && !visited[i - 1]) {
			continue;
		    }

		    //Mark the element visited.
		    visited[i] = true;
		    //Add it in child list.
		    cl.add(nums[i]);

		    findPermutations(nums, cl, visited);

		    //Backtracking : Remove the recently added element.
		    cl.remove(cl.size() - 1);
		    //Mark the element unvisited.
		    visited[i] = false;
		}
	    }
	}



## 31. Next Permutation

	class Solution {
	    public void nextPermutation(int[] arr) {
		int n = arr.length;

		int pivot = -1;
		for(int i=n-2; i>=0; --i){
		    if(arr[i]<arr[i+1]){
			pivot=i; 
			break;
		    }
		}
		// this means no valid index found to generate next higher permutation.
		if(pivot==-1){
		    reverseArr(arr, -1); // reversing whole arr
		    return;
		}

		int biggerInd= -1;
		for(int i=n-1; i>pivot; --i){
		    if(arr[i] > arr[pivot]){  // after pivot getting first greater value.
			biggerInd = i;
			break;
		    }
		}

		swap(arr, pivot, biggerInd); // swapping bigger val to its appropriate place.
		//since, pivot was the 1st smaller index, this means before pivot values must be greater & reversing is required to get the next higher permutation.
		reverseArr(arr, pivot);   

	    }

	    public void swap(int[] arr, int i, int j){
		int temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	    }

	    public void reverseArr(int[] arr, int ind){
		int l=ind+1;
		int r=arr.length-1;
		while(l<=r){
		    swap(arr, l, r);
		    l++; r--;
		}
	    }

	}



## 37. Sudoku Solver

	class Solution {
	    private boolean isSafe(char[][] board,int i,int j,int c){
		for (int p = 0; p < 9; p++) {
		    if(board[p][j]==c)
			return false;
		}
		for (int p = 0; p < 9; p++) {
		    if(board[i][p]==c)
			return false;
		}

		int row = i-i%3;
		int col = j-j%3;
		for(int p = row;p<row+3;p++){
		    for(int q = col;q<col+3;q++){
			if(board[p][q] == c){
			    return false;
			}
		    }
		}
		return true;
	    }
	    public boolean dfs(char[][] board){
		for(int i=0;i<9;i++){
		    for(int j=0;j<9;j++){
			if(board[i][j]=='.'){
			    for(char k='1';k<='9';k++){
				if(isSafe(board,i,j,k)){
				    board[i][j] = k;
				    if(dfs(board))
					return true;
				    else
					board[i][j]='.';
				}

			    }
			    // this is bcoz we are unable to generate a valid sudoko with all the possible explorations.
			    return false;
			}
		    }
		}
		return true;
	    }
	    public void solveSudoku(char[][] board) {
		if(board.length==0) {
		    return;
		}
		dfs(board);
	    }
	}



## 40. Combination Sum II

	class Solution {
	    List<List<Integer>> ans = new ArrayList<>();
	    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
		Arrays.sort(candidates);
		backtracking(candidates,target,0,new ArrayList<>());
		return ans;
	    }
	    public void backtracking(int [] candidates,int target,int rejected, ArrayList<Integer> list){
		if(target<0){
		    return;
		}
		if(target==0){
		    ans.add(new ArrayList<>(list));
		    return;
		}
		for(int i=rejected;i<candidates.length;i++){
		    if(i>rejected && candidates[i]==candidates[i-1]){
			continue;
		    }
		    list.add(candidates[i]);
		    backtracking(candidates,target-candidates[i],i+1,list);
		    list.remove(list.size()-1);
		}
	    }
	}






## 295. Find Median from Data Stream

	class MedianFinder {

	    private Queue<Integer> largeNumMinHeap;                     // Min-Heap storing larger numbers, will store at most k ints
	    private Queue<Integer> smallNumsMaxHeap;                    // Max-Heap storing smaller numbers, will store at most k + 1 ints

	    public MedianFinder() {
		this.largeNumMinHeap = new PriorityQueue<Integer>((a, b) -> Integer.compare(a, b));
		this.smallNumsMaxHeap = new PriorityQueue<Integer>((a,b) -> Integer.compare(b, a));
	    }
	    public void addNum(int num) {
		smallNumsMaxHeap.offer(num);
		largeNumMinHeap.offer(smallNumsMaxHeap.poll());
		if (smallNumsMaxHeap.size() < largeNumMinHeap.size())
		{
		    smallNumsMaxHeap.offer(largeNumMinHeap.poll());
		}
	    }
	    public double findMedian() {
		if (smallNumsMaxHeap.size() > largeNumMinHeap.size())
		{
		    return smallNumsMaxHeap.peek();
		}
		else
		{
		    return (smallNumsMaxHeap.peek() + largeNumMinHeap.peek()) / 2.0;
		}
	    }
	}


239. Sliding Window Maximum

	class Solution {
	    public int[] maxSlidingWindow(int[] nums, int k) {
		Deque<Integer> dq = new LinkedList();
		int n = nums.length;
		int ans[] = new int[n-k+1];

		for(int l = 0, r = 0; r < n; r++)
		{
		    // Left boundary of the window.
		    l = r - k + 1;
		    int curr = nums[r];

		    // Check recently added index and remove it if less than current element,
		    // to maintain the decreasing order in the Queue(deck).
		    while(!dq.isEmpty() && nums[dq.peekLast()] < curr)
			dq.pollLast();
		    dq.addLast(r);

		    // Remove the left index from queue, if it is outside the window.
		    if(!dq.isEmpty() && l > dq.peekFirst())
			dq.pollFirst();

		    if(r+1 >= k)
		    {
			ans[l] = nums[dq.peekFirst()];
			l++;
		    }
		}

		return ans;
	    }
	}


## 124. Binary Tree Maximum Path Sum

	class Solution {

	    public int maxPathSum(TreeNode root) {

		 int[] maxSum = new int[1];
		 maxSum[0] = Integer.MIN_VALUE; 

		findPathSum(root,maxSum);
		return maxSum[0];
	    }
	    private int findPathSum(TreeNode root,int[] maxSum) {
		if(root == null) return 0;

		int left = Math.max(0 , findPathSum(root.left , maxSum));
		int right = Math.max(0 , findPathSum(root.right , maxSum));

		maxSum[0] = Math.max(maxSum[0] , left + right + root.val);

		return root.val + Math.max(left , right);
	    }
	}



## 51. N-Queens

	class Solution {
	  public List<List<String>> solveNQueens(int n) {
	    boolean[][] board = new boolean[n][n];
	    List<List<String>> answer = new ArrayList<>();
	    queens(board,0,answer);
	    return answer;
	}
	void queens(boolean[][] board,int row,List<List<String>> answer){

	    if(row==board.length){
		List<String> res = insert(board);
		answer.add(res);
		return;
	    }

	    // Placing Queens and Checking for every row and column
	    for(int col=0;col<board.length;col++){
		//place queen if Safe
		if(isSafe(board,row,col)){
		    board[row][col] = true;
		    queens(board,row+1,answer); //Recursive call
		    board[row][col] = false;    // Backtrack
		}
	    }
	}

	boolean isSafe(boolean[][] board, int row ,int col){

	    // Vertical row
	    for(int i=0;i<row;i++){
		if(board[i][col])
		    return false;
	    }

	    // Left diagonal
	    int maxLeft = Math.min(row,col);
	    for(int i=1;i<=maxLeft;i++){
		if(board[row-i][col-i]){
		    return false;
		}
	    }

	    // Right diagonal
	    int maxRight = Math.min(row,board.length-col-1);
	    for(int i=1;i<=maxRight;i++){
		if(board[row-i][col+i]){
		    return false;
		}
	    }
	    return true;
	}

	  List<String> insert(boolean[][] board){

	    List<String> ans = new ArrayList<>();
	    for(boolean[] row: board){
		String S = "";
		for(boolean element:row){
		    if(element){
			S += "Q";
		    }
		    else
			S += ".";
		}
		ans.add(S);
	    }
	      return ans;
	}
	}



## 32. Longest Valid Parentheses

	class Solution {
	public int longestValidParentheses(String s) {
		if(s==null){
		    return 0;
		}
		Stack<Integer> stk = new Stack();
		int result = 0;
		int lastI = -1; //it will remain -1 for ()() , (()). Will change if extra closing ')' is encountered e.g. ) ()()
	    // note we are pushing only ( in the stack
		for(int i=0; i<s.length(); i++) {
		    if(s.charAt(i)=='(') {
			stk.push(i);      
		    } 

		    else {

			if(!stk.isEmpty()) { // there is matching '(' in stack
			    stk.pop();  //remove matching '('
			    //find valid string start based on stack emptiness

			    if(!stk.isEmpty()) { 
				//there are unclosed '(' other than the one popped '(' in prev line e.g. for (()  , ( is still in stack
				result = Math.max(result, i-stk.peek());
				//so valid str len will start after unclosed '(' which is still in stack 
			    }
			    else 
			    {
				result = Math.max(result, i-lastI); //stack is empty now, use lastI to find the begining which is either after 0 for which lastI = -1 for it or  after extra ) e.g. ()() -> lastI =-1 or )()() -> lastI = 0
			    }    

			} 

			else //(while stack is empty and we get )
			{ 
			    // there is extra ) that has no matching '(', preserve the lastI
			    // this is case for like )()()
			    lastI = i;
			}
		    }
		}
		return result;
	    }
	 } 



## 25. Reverse Nodes in k-Group

	class Solution {
	    public ListNode reverseKGroup(ListNode head, int k) {
	       if(k==1){
		   return head;
	       } 
		ListNode left=head;
		ListNode right=head;
		ListNode prev=null;
		int j=0;
		while(right!=null && right.next!=null){
		int i=1;
		    while(i<k &&right!=null){
		     right=right.next; 
		    i++;
		   }
		   if(right!=null){
		       reverse(left,right,prev);
		       j++;
		   }else{
		       break;
		   }
		    if(j==1){
			head=right;
		    }
		    if(prev!=null){
		   prev.next=right;
		    }
		   prev=left;
		   right=left.next;
		   left=left.next;
		} 
		return head;
	    }
	    public void reverse(ListNode left,ListNode right,ListNode prev){
		ListNode temp=left;
		ListNode last=prev;
	      while(prev!=right){

	       ListNode frwd=left.next;
		left.next=prev;
		prev=left;
		left=frwd;
	     }
	      temp.next=left;
		if(last!=null){
		last.next=right;
		} 
	    }
	}



## 23. Merge k Sorted Lists

	class Solution {
	    public ListNode mergeKLists(ListNode[] lists) {

	/*
	Intuition behind the approach is to initialise a min heap and then add every element of the list to the min heap.
	Since it is a min heap, therefore root will be the smallest element and we can extract it and add to the new list.


	*/
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>();
		//ading everything to the new priority queue
		for(ListNode ln : lists) {
		    while(ln != null) {
			pq.add(ln.val);
			ln = ln.next;
		    }
		}
		//list to store the sorted list
		ListNode head = new ListNode(0);
		ListNode h = head;

		while(!pq.isEmpty()) {
		    ListNode t = new ListNode(pq.poll());
		    h.next = t;
		    h = h.next;
		}

		return head.next;

	    }
	}






## 22. Generate Parentheses

	class Solution {
	    public List<String> generateParenthesis(int n) {
		    List<String> res=new ArrayList<>();
		    if(n<1)
			return res;
		    solve(res, n, 0, 0, "");
		    return res;
		}

		void solve(List<String> res, int n, int open, int close, String str){
		    if(open==close && close==n){
			res.add(str);
			return;
		    }
		    if(open<n)
			solve(res, n, open+1, close, str+"(");
		    if(close<open)
			solve(res, n, open, close+1, str+")");
		}
	}



## 24. Swap Nodes in Pairs

	class Solution {

		public ListNode swapPairs(ListNode head) {
			if (head == null || head.next == null)
				return head;
			ListNode dummy = new ListNode(0);
			ListNode prev = dummy;
			dummy.next = head;
			ListNode curr = head;
			ListNode forw = null;
			while (curr != null && curr.next != null) {
				forw = curr.next;
				curr.next = forw.next;
				prev.next = forw;
				forw.next = curr;

				prev = prev.next.next;
				curr = curr.next;

			}
			return dummy.next;
		}

	}



## 73. Set Matrix Zeroes

	class Solution {
	    public void setZeroes(int[][] matrix) {
		int m = matrix.length;
		int n = matrix[0].length;

		// Store the rows and columns number where there is a 0.
		List<Integer> rows = new ArrayList<> ();
		List<Integer> cols = new ArrayList<> ();

		for(int i = 0; i < m; i++){
		    for(int j = 0; j < n; j++){

			// Check if whether paricular column and row 
			// has 0 or not. If it has then add row and column
			// to the corresponding list.
			if(matrix[i][j] == 0){
			    rows.add(i);
			    cols.add(j);
			}
		    }
		}

		// Traverse through rows and columns if they stored
		// any value that means this particular row and column
		// has 0.
		for(int i = 0; i < rows.size(); i++){
		    int row = rows.get(i);
		    int col = cols.get(i);

		    // Put 0 to all the cell of the row.
		    for(int j = 0; j < n; j++){
			matrix[row][j] = 0;
		    }

		    // Put 0 to all the cell of the column.
		    for(int k = 0; k < m; k++){
			matrix[k][col] = 0;
		    }
		}
	    }
	}



## 78. Subsets

	class Solution {

	    void solve(int[] nums,int ind,List<List<Integer>> ans,List<Integer> temp){


		if(ind >= nums.length){

		    // remember to copy the arraylist in an another array list before placing it in             the ans else it will not be included in the final ans
		    List<Integer> ls = new ArrayList<>();
		    for(Integer a: temp){
			ls.add(a);
		    }
		    ans.add(ls);
		    return;
		}

		temp.add(nums[ind]);
		solve(nums,ind+1,ans,temp);
		temp.remove(temp.size()-1);  
		solve(nums,ind+1,ans,temp);
	    }

	    public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> ans = new ArrayList<>();
		List<Integer> temp = new ArrayList<>();
		solve(nums,0,ans,temp);
		return ans;
	    }
	}



## 78. Subsets

	class Solution {
	    public boolean check(char[][] board,String s,int idx,int i,int j,boolean[][] visited){
		if(idx==s.length()){
		    return true;
		}
		if(i<0 || j<0 || i>=board.length || j>=board[0].length || visited[i][j]==true || s.charAt(idx)!=board[i][j]) return false;
		visited[i][j] = true;
		if(check(board,s,idx+1,i+1,j,visited)) return true;
		if(check(board,s,idx+1,i,j+1,visited)) return true;
		if(check(board,s,idx+1,i,j-1,visited)) return true;
		if(check(board,s,idx+1,i-1,j,visited)) return true;
		visited[i][j] = false;
		return false;
	    }
	    public boolean exist(char[][] board, String word) {
		for(int i=0;i<board.length;i++){
		    for(int j=0;j<board[0].length;j++){
			if(word.charAt(0)==board[i][j]){
			    boolean[][] visited = new boolean[board.length][board[0].length];
			    if(check(board,word,0,i,j,visited)) return true;
			}
		    }
		}
		return false;    
	    }
	}



## 105. Construct Binary Tree from Preorder and Inorder Traversal

	class Solution {
	    int preStart;
	    TreeNode buildTreeHelper(int[]preorder, int[]inorder, int is, int ie){
		if(is > ie){
		    return null;
		}
		TreeNode root = new TreeNode(preorder[preStart]);
		preStart++;
		if(is == ie){
		    return root;
		}
		int rootIndex = -1;
		for(int i = is; i<=ie; i++){
		    if(inorder[i] == root.val){
			rootIndex = i;
			break;
		    }
		}
		root.left = buildTreeHelper(preorder,inorder,is,rootIndex-1);
		root.right = buildTreeHelper(preorder,inorder,rootIndex+1,ie);
		return root;
	    }
	    public TreeNode buildTree(int[] preorder, int[] inorder) {
		preStart = 0;
		return buildTreeHelper(preorder,inorder,0,inorder.length-1);
	    }
	}




## 128. Longest Consecutive Sequence

	class Solution {
	public int longestConsecutive(int[] nums)
	{
	if(nums.length<1)
	return 0;

	    Arrays.sort(nums);

	    int j=0;
	    int max=0;
	    for(int i=0;i<nums.length-1;i++)
	    {
		int x = nums[i+1]-nums[i];
		if(x==1)
		    j++;
		else if(x != 1 && x !=0)
		{
		    j=0;
		}
		max = Math.max(max,j);     
	    }
	    return max+1;
	}

	}



## 131. Palindrome Partitioning

	class Solution {
	    public List<List<String>> partition(String s) {
		List<List<String>> ans=new ArrayList<>();
		partition(s,new ArrayList<>(),ans);
		return ans;
	    }

	    public void partition (String s,List<String> list, List<List<String>> ans){
		//Base case : when we are done traversing the whole string
		if(s.length()==0){
		    ans.add(new ArrayList<String>(list));
		    return ;
		}

		for(int i=0;i<s.length();i++){
		    String prefix=s.substring(0,i+1);
		    String ros=s.substring(i+1);
		    if(isPallin(prefix)){
			list.add(prefix);
			partition(ros,list,ans);
			list.remove(list.size()-1); //backtracking 
		    }
		}
	    }

	    public boolean isPallin(String x){
		int i=0,j=x.length()-1;
		while(i<j){
		    if(x.charAt(i)==x.charAt(j)){
			i++; j--;
		    }
		    else
			return false;
		}
		return true;
	    }
	}



## 146. LRU Cache

	class LRUCache {
	    Node head = new Node(0,0); 
	    Node tail = new Node(0,0);
	    Map<Integer,Node> map = new HashMap<>();
	    int limit=0;
	    public LRUCache(int capacity) {
		limit = capacity;
		head.next=tail;
		tail.prev=head;
	    }

	    public int get(int key) {
		if(map.containsKey(key)){
		    Node node = map.get(key);
		    remove(node);
		    insert(node);
		    return node.data;
		}else
		return -1;
	    }

	    public void put(int key, int value) {
		if(map.containsKey(key))
		remove(map.get(key));
		if(map.size()==limit)
		remove(tail.prev);
		insert(new Node(key,value));
	    }
	    public void remove(Node node){
		map.remove(node.key);
		node.prev.next = node.next;
		node.next.prev = node.prev;
	    }
	    public void insert(Node node){
		map.put(node.key, node);
		Node newNode = head.next;
		head.next = node;
		node.prev=head;
		newNode.prev=node; 
		node.next=newNode;
	    }
	}
	class Node{
	    int data,key;
	    Node next;
	    Node prev;
	    Node(int key,int data){
		this.data=data;
		this.key= key;
		this.next = null;
		this.prev=null;
	    }
	}




## 200. Number of Islands

	class Solution {
	    public int numIslands(char[][] grid) {

		int m = grid.length;
		int n = grid[0].length;
		int ans = 0;
		for(int i=0;i<m;i++){
		    for(int j=0;j<n;j++){
			if(grid[i][j]=='1'){
			    dfs(grid,i,j,m,n);
			    ans++;
			}
		    }
		}
		return ans;
	    }
	    static void dfs(char[][] ch, int i, int j, int m, int n){
		if(j<0 || i<0 || i>=m || j>=n || ch[i][j]!='1'){
		    return;
		}
		ch[i][j] = '2';
		dfs(ch,i,j-1,m,n);
		dfs(ch,i-1,j,m,n);
		dfs(ch,i,j+1,m,n);
		dfs(ch,i+1,j,m,n);
	    }

	}



## 207. Course Schedule

	class Solution {
	    public boolean canFinish(int numCourses, int[][] prerequisites) {
		if(prerequisites == null || prerequisites.length == 0) return true;

		//generate map <course_id> - <list of pre requisted courses>
		Map<Integer, List<Integer>> pre_req_courses = new HashMap<>();
		for(int i = 0; i < prerequisites.length; i++){
		    int[] pre_req = prerequisites[i];
		    if(!pre_req_courses.containsKey(pre_req[0]))
			pre_req_courses.put(pre_req[0], new ArrayList<>());
		    pre_req_courses.get(pre_req[0]).add(pre_req[1]);
		}

		int[] visited = new int[numCourses];
		//0 - means hasn't been visited
		//1 - permanent mark - means has been visited and no cycle detected
		//2 - temporary mark - means it is during the DFS cycle(on the path), if it is visited again - means a cycle detected

		for (int i = 0; i < numCourses; i++) {
		    if(visited[i] == 0){//this course has been visited, do a dfs search to check if there is a cycle
			if(isCyclic(visited, i, pre_req_courses)) return false;
		    } 
		}

		return true;
	    }

	    private boolean isCyclic(int[] visited, int i, Map<Integer, List<Integer>> pre_req_courses){
		visited[i] = 2; //temporary mark - means it is currently in a DFS search

		if(pre_req_courses.containsKey(i)){//if this course i has some pre required courses to finish first            
		    List<Integer> must_finish_before = pre_req_courses.get(i);//get the list of pre required courses
		    for(int prereq_course_id : must_finish_before){//for each course in the list
			if(visited[prereq_course_id] == 2) return true;//if this course has been visited, then a cycle is detected
			if(visited[prereq_course_id] == 0) {//hasn't been visited, do a DFS search
			    if(isCyclic(visited, prereq_course_id, pre_req_courses)) return true;
			}
		    }
		}

		visited[i] = 1; //permanent mark - this course has been visited, no cycle detected
		return false;
	    }
	}



## 394. Decode String

	class Solution {
	    public String decodeString(String s) {
		ArrayDeque<Integer> intStack = new ArrayDeque<>();
		ArrayDeque<String> strStack = new ArrayDeque<>();
		StringBuilder curr = new StringBuilder();
		int count = 0;
		for (char ch : s.toCharArray()) {
		    if ('0' <= ch && ch <= '9') {
			count = count * 10 + ch - '0';
			continue;
		    }
		    if (ch == '[') {
			intStack.push(count);
			strStack.push(curr.toString());
			curr = new StringBuilder();
			count = 0;
			continue;
		    }
		    if (ch == ']') {
			String PrevStr = strStack.pop();
			int PrevInt = intStack.pop();
			PrevStr += (curr.toString().repeat(PrevInt));
			curr = new StringBuilder(PrevStr);
			continue;
		    }
		    curr.append(ch);
		}
		return curr.toString();
	    }
	}



## 647. Palindromic Substrings

	class Solution {

		int count=0;
		int result=0;

		int countSubstrings(String s) {

					if (s == null || s.length() == 0) 
						return 0;

			for (int i = 0; i < s.length(); i++) { 
				extendPalindrome(s, i, i); 
				extendPalindrome(s, i, i + 1); 
			}

			return count;
		}

		public void extendPalindrome(String s, int leftside, int rightside) {
			while (leftside >=0 && rightside < s.length() && s.charAt(leftside) == s.charAt(rightside)) {
				count++; 
				leftside--; 
				rightside++;
			}
		}
	}



## 763. Partition Labels

	class Solution {
	    public List<Integer> partitionLabels(String s) {
		Map<Character, Integer> map = new HashMap<>();
		// filling impact of character's
		for(int i = 0; i < s.length(); i++){
		    char ch = s.charAt(i);
		    map.put(ch, i);
		}
		List<Integer> res = new ArrayList<>();
		int prev = -1;
		int max = 0;

		for(int i = 0; i < s.length(); i++){
		    char ch = s.charAt(i);
		    max = Math.max(max, map.get(ch));
		    if(max == i){
			// partition time
			res.add(max - prev);
			prev = max;
		    }
		}
		return res;
	    }
	}





