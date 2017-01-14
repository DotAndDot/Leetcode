+ 344 Reverse String


+ 345 Reverse Vowels of a String


+ 20 Valid Parentheses

  1. 栈
  2. ​

  ```c++
  bool isValid(string s) {
      int top = -1;
      for (int i = 0; i < s.size(); i++) {
          if ((top == -1) && (s[i] == ')' || s[i] == '}' || s[i] == ']')) {
              return false;
          }
          switch (s[i]) {
              case ')': case '}': case ']':
                  if (abs(s[i] - s[top]) < 3) top--;
                  else return false;
                  break;
              default:
                  s[++top] = s[i];
                  break;
          }
      }
      return top == -1;
  }
  ```


+ 28 Implement strStr()

  kmp算法：

  + 求next数组

    "部分匹配值"就是"前缀"和"后缀"的最长的共有元素的长度。以"ABCDABD"为例，

     　　－　"A"的前缀和后缀都为空集，共有元素的长度为0；
     　　－　"AB"的前缀为[A]，后缀为[B]，共有元素的长度为0；
     　　－　"ABC"的前缀为[A, AB]，后缀为[BC, C]，共有元素的长度0；
     　　－　"ABCD"的前缀为[A, AB, ABC]，后缀为[BCD, CD, D]，共有元素的长度为0；
     　　－　"ABCDA"的前缀为[A, AB, ABC, ABCD]，后缀为[BCDA, CDA, DA, A]，共有元素为"A"，长度为1；
     　　－　"ABCDAB"的前缀为[A, AB, ABC, ABCD, ABCDA]，后缀为[BCDAB, CDAB, DAB, AB, B]，共有元素为"AB"，长度为2；
     　　－　"ABCDABD"的前缀为[A, AB, ABC, ABCD, ABCDA, ABCDAB]，后缀为[BCDABD, CDABD, DABD, ABD, BD, D]，共有元素的长度为0。

  + 求next数组

    ```c++
    void makeNext(const char P[],int next[])
    {
        int q,k;//q:模版字符串下标；k:最大前后缀长度
        int m = strlen(P);//模版字符串长度
        next[0] = 0;//模版字符串的第一个字符的最大前后缀长度为0
        for (q = 1,k = 0; q < m; ++q)//for循环，从第二个字符开始，依次计算每一个字符对应的next值
        {
            while(k > 0 && P[q] != P[k])//递归的求出P[0]···P[q]的最大的相同的前后缀长度k
                k = next[k-1];          //不理解没关系看下面的分析，这个while循环是整段代码的精髓所在，确实不好理解  
            if (P[q] == P[k])//如果相等，那么最大相同前后缀长度加1
            {
                k++;
            }
            next[q] = k;
        }
    }
    ```

    已知前一步计算时最大相同的前后缀长度为k（k>0），即P[0]···P[k-1]；

    　　此时比较第k项P[k]与P[q],如图1所示

    　　如果P[K]等于P[q]，那么很简单跳出while循环;

    　　**关键！关键有木有！关键如果不等呢？？？**那么我们应该利用已经得到的next[0]···next[k-1]来**求P[0]···P[k-1]这个子串中最大相同前后缀**，可能有同学要问了——为什么要求P[0]···P[k-1]的最大相同前后缀呢？？？是啊！为什么呢？ **原因**在于P[k]已经和P[q]失配了，而且P[q-k] ··· P[q-1]又与P[0] ···P[k-1]相同，看来P[0]···P[k-1]这么长的子串是用不了了，那么我要找个同样也是P[0]打头、P[k-1]结尾的子串即P[0]···P[j-1](j==next[k-1])，看看它的下一项P[j]是否能和P[q]匹配。如图2所示

    my:

    ```c++
    class Solution {
    public:
        void makenext(string needle, int next[]){
            next[0] = 0;
            for(int i = 1, k = 0; i < needle.size(); i++){
                while(k > 0 && needle[i] != needle[k])
                    k = next[k - 1];
                if(needle[i] == needle[k])
                    k++;
                next[i] = k;
            }
        }
        int strStr(string haystack, string needle) {
            if(needle == "")    return 0;
            int next[needle.size()] = {0};
            makenext(needle, next);
            int n = needle.size();
            for(int i = 0, j = 0;i < haystack.size(); i++){
                while(j > 0 && haystack[i] != needle[j])
                    j = next[j - 1];
                if(haystack[i] == needle[j])
                    j++;
                if(j == n)
                    return i - j + 1;
            }
            return -1;
        }
    };
    ```

+ 58 Length of Last Word

  需要考虑s[s.size() - 1]处为空格的情况


+ 8 String to Integer (atoi)

  int myAtoi(string str)：需要转换的字符串为str中从第一个不是空格处开始至下个空格或结尾的子串；int的范围为-2147483648 - 2147483647，需要考虑转换的整数超过int范围的情况，可使用long类型解决这一问题

+ 227 Basic Calculator II

  ​1.使用栈计算乘除法，最后遍历完成加减法

  ​2.二叉树？

+ 22 Generate Parentheses

  递归；

  卡特兰数:

  令h(0)=1,h(1)=1，catalan数满足递推式：

  **h(n)= h(0)\*h(n-1)+h(1)*h(n-2) + ... + h(n-1)h(0) (n>=2)**

  递推关系的解为：

  **h(n)=C(2n,n)/(n+1) (n=0,1,2,...)**

  ​出栈次序

  一个栈(无穷大)的进栈序列为1，2，3，…，n，有多少个不同的出栈序列

  首先，我们设f（n）=序列个数为n的出栈序列种数。（我们假定，最后出栈的元素为k，显然，k取不同值时的情况是相互独立的，也就是求出每种k最后出栈的情况数后可用加法原则，由于k最后出栈，因此，在k入栈之前，比k小的值均出栈，此处情况有f(k-1)种，而之后比k大的值入栈，且都在k之前出栈，因此有f(n-k)种方式，由于比k小和比k大的值入栈出栈情况是相互独立的，此处可用乘法原则，f(n-k)*f(k-1)种，求和便是Catalan递归式

  ```c++
  class Solution {
  public:
      vector<string> generateParenthesis(int n) {
          vector<string> result;
          helper(result, n, n, "");
          return result;
      }
      void helper(vector<string>& result, int m, int n, string temp){
          if(m == 0 && n == 0){
              result.push_back(temp);
              return;
          }
          if(m>0)
              helper(result, m-1, n, temp+'(');
          if(m<n)
              helper(result, m, n-1, temp+')');
      }
  };
  ```

+ 3 Longest Substring Without Repeating Characters

  在遍历字符串的同时使用hash表记录每个字符上一次出现的位置，找出任意两个相邻的相同字符之间最大的长度

  ```c++
   int loc[256];
   memset(loc, -1, sizeof(loc));
   for(int i = 0; i < s.size(); i++){
   	loc[s[i]] = i;
   }
  ```



+ 91 Decode Ways

  动态规划：状态   、    状态转移方程


+ 338 Counting Bits

  状态转移方程：dp[i] = dp[2^n] + dp[i - 2^n]


+ 213 House Robber II

  House Robber I的升级版. 因为第一个element 和最后一个element不能同时出现. 则分两次call House Robber I. case 1: 不包括最后一个element. case 2: 不包括第一个element.

  两者的最大值即为全局最大值

+ 63 Unique Paths II

  ```c++
  class Solution {
  public:
      int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
          int m = obstacleGrid.size(), n = obstacleGrid[0].size();
          int dp[m][n] = {0};
          for(int i = 0; i < m; i++)
              for(int j = 0; j < n; j++)
                  dp[i][j] = 0;
          if(obstacleGrid[0][0] == 0)
              dp[0][0] = 1;
          for(int i = 0; i < m; i++){
              for(int j = 0; j < n; j++){
                  if(obstacleGrid[i][j] == 0){
                      if(i == 0 && j > 0)
                          dp[i][j] = dp[i][j - 1];
                      else if(i > 0 && j == 0)
                          dp[i][j] = dp[i - 1][j];
                      else if(i > 0 && j > 0)
                          dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                  }
              }
          }
          return dp[m-1][n-1];
      }
  };
  ```

+ 343 Integer Break

  将整数拆解为3、2相加时所得的乘积最大，所以只用考虑dp[i-2] * 2 和 dp[i-3] * 3中最大的即可


+ 91 Submission Details

  若s[i] >'0', 则s[i] 的 decode ways 包含 s[i-1] 的；

  若s[i - 1] > '0'且s[i-1]s[i]构成的2位数小于等于26时， 则s[i] 的 decode ways 包含 s[i-2] 的；

  在确定初始状态时需要注意s[1]为0的情况，即s[i-1]s[i]构成的2位数大于26


+ 71 Simplify Path

  用栈解决，/..出栈，/.和/不操作，其他情况进栈

  边界条件："/../" => "/"  ;   "/home//foo/" => "/home/foo"

+ 5 Longest Palindromic Substring

  1 Manacher算法：http://www.cnblogs.com/bitzhuwei/p/Longest-Palindromic-Substring-Part-II.html

  2 dp：类似于lcs的解法，数组flag\[i][j]记录s从i到j是不是回文

  首先初始化，i==j时，flag\[i][j]=true，这是因为s\[i][i]是单字符的回文，当i>j时，为true，是因为有可能出现flag\[2][1]这种情况，比如bcaa，当计算s从2到3的时候，s[2]==s[3]，这时就要计算s[2+1] ?= s[3-1]，总的来说，当i>j时置为true，就是为了考虑j=i+1这种情况。

  接着比较s[i] ?= s[j]，如果成立，那么flag\[i][j] = flag\[i+1][j-1]，否则直接flag\[i][j]=false


+ 49 Group Anagrams

  ```c++
  class Solution {
  public:
      vector<vector<string>> groupAnagrams(vector<string>& strs) {
          unordered_map<string, vector<string>> hasht;
          for(auto a : strs){
              string t = a;
              sort(t.begin(), t.end());
              hasht[t].push_back(a);
          }
          vector<vector<string>> res;
          for(auto a : hasht){
              res.push_back(a.second);
          }
          return res;
      }
  };
  ```

+ 76 Minimum Window Substring

  用need记录t中各字符需要出现的次数，found为当前出现过的次数，count为t中字符需要出现总次数，使用双指针left、right，left和right同时从最左端开始，先移动right，当left 到 right间包括t中所有字符后，left向right移动；当left和right间不包括t中所以字符后，right移动；如此循环。

  ```c++
  class Solution {
  public:
      string minWindow(string s, string t) {
          int minl = 0, minr = 0, len = INT_MAX, left = 0, right = 0, count = t.size();
          vector<int> need(128, 0);
          vector<int> found(128,0);
          for(auto a : t){
              need[a]++;
          }
          while(left <= s.size() - t.size() && right < s.size()){
              if(count != 0){
                  while(right < s.size() && count > 0){
                      found[s[right]]++;
                      if(need[s[right]] >= found[s[right]]){
                          count--;
                      }
                      right++;
                  }
              }
              if(count == 0){
                  while(left < right && count == 0){
                      if(right - left < len){
                          minl = left;
                          minr = right - 1;
                          len = right -left;
                      }
                      found[s[left]]--;
                      if(need[s[left]] > found[s[left]]){
                          count++;
                      }
                      left++;
                  }
              }
          }
          return len == INT_MAX ? "" : s.substr(minl, len);
      }
  };
  ```


+ 65 Valid Number

  有限状态自动机：

  有限状态机的状态转移过程：

  起始为0：

  　　当输入空格时，状态仍为0，

  　　输入为符号时，状态转为3，3的转换和0是一样的，除了不能再接受符号，故在0的状态的基础上，把接受符号置为-1；

  　　当输入为数字时，状态转为1, 状态1的转换在于无法再接受符号，可以接受空格，数字，点，指数；状态1为合法的结束状态；

  　　当输入为点时，状态转为2，状态2必须再接受数字，接受其他均为非法；

  　　当输入为指数时，非法；

  状态1：

  　　接受数字时仍转为状态1，

  　　接受点时，转为状态4，可以接受空格，数字，指数，状态4为合法的结束状态，

  　　接受指数时，转为状态5，可以接受符号，数字，不能再接受点，因为指数必须为整数，而且必须再接受数字；

  状态2：

  　　接受数字转为状态4；

  状态3：

  　　和0一样，只是不能接受符号；

  状态4：

  　　接受空格，合法接受；

  　　接受数字，仍为状态4；

  　　接受指数，转为状态5，

  状态5：

  　　接受符号，转为状态6，状态6和状态5一样，只是不能再接受符号，

  　　接受数字，转为状态7，状态7只能接受空格或数字；状态7为合法的结束状态；

  状态6：

  　　只能接受数字，转为状态7；

  状态7：

  　　接受空格，转为状态8，状态7为合法的结束状态；

  　　接受数字，仍为状态7；

  状态8：

  　　接受空格，转为状态8，状态8为合法的结束状态；

  使用2维数组changesta作为状态转移表，changesta\[i][j]表示状态i在接收j输出后所转移的状态

+ 32 Longest Valid Parentheses

  动态规划：

  我的LTE

  ```c++
  class Solution {
  public:
      bool judge(vector<vector<bool>> &dp, int i, int j){
          for(int k = i + 1; k < j - 1; k++)
              if(dp[i][k] && dp[k + 1][j])
                  return true;
          return false;
      }
      int longestValidParentheses(string s) {
          vector<vector<bool>> dp(s.size(), vector<bool>(s.size(), false));
          int max = 0;
          for(int i = 0; i < s.size(); i++){
              for(int j = 0; j < s.size(); j++){
                  if(j - i == 1 && s[i] == '(' && s[j] == ')'){
                      dp[i][j] = true;
                      max = 2;
                  }
                  else
                      dp[i][j] == false;
              }
          }
          for(int j = 3; j < s.size(); j++){
              for(int i = j - 3; i >= 0 ; i--){
                  if(s[i] == '(' && s[j] == ')' && dp[i + 1][j - 1] == true)
                      dp[i][j] = true;
                  else if(judge(dp, i, j))
                      dp[i][j] = true;
                  else
                      dp[i][j] = false;
                  if(dp[i][j] == true && j - i + 1 > max)
                      max = j - i + 1;
              }
          }
          return max;
      }
  };
  ```

  bool型数组dp，dp\[i][j]为i到j的字符串是否是有效的，当s[i]为'(' ,s[j] 为')'时，若dp\[i + 1][j - 1] == true， 则dp\[i][j] = true;否则取 i < k < j - 1,判断是否存在k使，dp\[i][k] ，dp\[k + 1][j]都为true，有则dp\[i][j] = true；都不满足则dp\[i][j] = false；

  能ac的算法，只需遍历一遍字符串：

  这道题可以用一维动态规划逆向求解。假设输入括号表达式为String s，维护一个长度为s.length的一维数组dp[]，数组元素初始化为0。 dp[i]表示从s[i]到s[s.length - 1]包含s[i]的最长的有效匹配括号子串长度。则存在如下关系：

  - dp[s.length - 1] = 0;
  - i从n - 2 -> 0逆向求dp[]，并记录其最大值。若s[i] == '('，则在s中从i开始到s.length - 1计算dp[i]的值。这个计算分为两步，通过dp[i + 1]进行的（注意dp[i + 1]已经在上一步求解）：在s中寻找从i + 1开始的有效括号匹配子串长度，即dp[i + 1]，跳过这段有效的括号子串，查看下一个字符，其下标为j = i + 1 + dp[i + 1]。若j没有越界，并且s[j] == ‘)’，则s[i ... j]为有效括号匹配，dp[i] =dp[i + 1] + 2。在求得了s[i ... j]的有效匹配长度之后，若j + 1没有越界，则dp[i]的值还要加上从j + 1开始的最长有效匹配，即dp[j + 1]。

  同理可以从0开始找

+ 336 Palindrome Pairs


​	用unordered_map存储各字符串出现的位置，遍历words，先判断反转后的字符串是否存在，然后分两种情况判断翻转后字符串字串；1.判断前半部分是否是回文，若是则判断剩下的部分是否存在，若存在则后半部分和原字符串组成回文串；2.判断后半部分是否是回文，若是则判断剩下部分是否存在，若存在则前半部分和原字符串组成回文

+ 88 Merge Sorted Array

  从后向前归并

+ 169 Majority Element

  每找出两个不同的element，则成对删除。最终剩下的一定就是所求的。

  可扩展到⌊ n/k ⌋的情况，每k个不同的element进行成对删除。

+ 189 Rotate Array

  1.先将数组reverse，然后再分别reverse数组的0 ~ k-1位和k ~ 最后一位

+ 1 Two Sum

  哈希表

+ 119 Pascal's Triangle II

  ```c++
  vector<int> getRow(int rowIndex) {
          vector<int> res;
          while(rowIndex >= 0){
              if(res.size() < 2)
                  res.push_back(1);
              else{
                  res.push_back(1);
                  for(int i = res.size() - 2; i >= 1; i--){
                      res[i] = res[i - 1] + res[i];
                  }
              }
              rowIndex--;
          }
          return res;
  ```

+ 162 Find Peak Element

  ​题目要求时间复杂度为O(logN)，logN时间的题目一般都是Heap，二叉树，分治法，二分搜索这些很“二”解法。这题是找特定元素，基本锁定二分搜索法。我们先取中点，由题意可知因为两端都是负无穷，有上坡就必定有一个峰，我们看中点的左右两边大小，如果向左是上坡，就抛弃右边，如果向右是上坡，就抛弃左边。直到两边都小于中间，就是峰了。


+  396 Rotate Function

   找出f(k) 和 f(k+1)的关系

+  153 Find Minimum in Rotated Sorted Array
              二分查找，当中点比最右边的点大的时候，说明最小值在中点右侧，抛弃左边；反正抛弃右边

   ```c++
     int findMin(vector<int>& nums) {
       int left = 0, right = nums.size() - 1;
       int mid = (left + right) / 2;
       if(nums.size() == 1)
         return nums[0];
       for(;left < right; mid = (left + right) / 2){
         if(nums[left] < nums[right]) return nums[left];
         else if(left + 1 == right)  return nums[left] < nums[right] ? nums[left] : nums[right];
         if(nums[mid] > nums[right])
           left = mid;
         else
           right = mid;
       }
       return nums[mid];  
     }
   ```

+  152 Maximum Product Subarray
             超时的算法：

   ```c++
     int maxProduct(vector<int>& nums) {
       if(nums.size() == 0)
         return 0;
       int len = nums.size();
       vector<vector<int>> dp(len, vector<int>(len));
       int max = nums[0];
       for(int i = 0; i < len; i++){
         dp[i][i] = nums[i];
         if(dp[i][i] > max)
           max = dp[i][i];
       }

       for(int i = 0; i < len; i++){
         for(int j = i + 1; j < len; j++){
           dp[i][j] = dp[i][j - 1] * nums[j];
           if(dp[i][j] > max)
             max = dp[i][j];
         }
       }
       return max;
     }
   ```

             其实子数组乘积最大值的可能性为：累乘的最大值碰到了一个正数；或者，累乘的最小值（负数），碰到了一个负数。所以每次要保存累乘的最大（正数）和最小值（负数）。同时还有一个选择起点的逻辑，如果之前的最大和最小值同当前元素相乘之后，没有当前元素大（或小）那么当前元素就可作为新的起点。例如，前一个元素为0的情况，{1,0,9,2}，到9的时候9应该作为一个最大值，也就是新的起点，{1,0,-9,-2}也是同样道理，-9比当前最小值还小，所以更新为当前最小值。

   ```c++
     int findmax(int a, int b){
       return a > b ? a : b;
     }
     int findmin(int a, int b){
       return a > b ? b : a;
     }
     int maxProduct(vector<int>& nums) {
       if(nums.size() == 0)
         return 0;
       int min = nums[0], max = nums[0], m = nums[0];
       for(int i = 1; i < nums.size(); i++){
         int a = min * nums[i];
         int b = max * nums[i];
         max = findmax(findmax(a, b), nums[i]);
         min = findmin(findmin(a, b), nums[i]);
         m = findmax(m, max);
       }
       return m;
     }
   ```

+  78 Subsets

	回溯：在循环中调用

   ```c++
     class Solution {
       public:
       vector<vector<int> > subsets(vector<int>& nums) {
         vector<vector<int> > ret;
         vector<int> sub;

         backtack(nums, sub, ret, 0);
         return ret;
       }

       void backtack(vector<int> &nums, vector<int> &sub,  vector<vector<int> > &ret, int index)
       {
         ret.push_back(sub);
         for (int i = index; i < nums.size(); i ++)
         {
           sub.push_back(nums[i]);
           backtack(nums, sub, ret, i + 1);
           sub.pop_back();
         }
       }
     };
   ```

+  96 Unique Binary Search Trees

           本题其实关键是递推过程的分析，n个点中每个点都可以作为root，当 i 作为root时，小于 i  的点都只能放在其左子树中，大于 i 的点只能放在右子树中，此时只需求出左、右子树各有多少种，二者相乘即为以 i 作为root时BST的总数。

   ```c++
     class Solution {
       public:
       int numTrees(int n) {
         int unique[n];
         unique[0] = 1;
         unique[1] = 1;
         for(int i = 2; i <= n; i++){
           unique[i] = 0;
           for(int j = 1; j <= i; j++){
             unique[i] += unique[j - 1] * unique[i - j];
           }
         }
         return unique[n];
       }
     };
   ```

+  106 Construct Binary Tree from Inorder and Postorder Traversal

         通过遍历序列还原二叉树，必须要包含中序遍历，根据终须遍历找到左右子树，递归构造原二叉树

   ```c++
                     /**
   * Definition for a binary tree node.
   * struct TreeNode {
   *     int val;
   *     TreeNode *left;
   *     TreeNode *right;
   *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
   * };
   */
   class Solution {
   public:
      void build(vector<int>& inorder, vector<int>& postorder, int istart, int iend, int pstart, int pend, TreeNode*& p){
          TreeNode* tmp = new TreeNode(postorder[pend]);
          p = tmp;
          int i;
          for(i = istart; i <= iend; i++){
              if(inorder[i] == postorder[pend])
                  break;
          }
          if(istart <= i - 1){
              build(inorder, postorder, istart, i - 1, pstart, pstart + (i - 1) - istart, tmp->left);
          }
          if(i + 1 <= iend){
          	build(inorder, postorder, i + 1, iend, pstart + i - istart, pend - 1, tmp->right);
   		}
      }
      TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
          TreeNode* p = NULL;
          if(inorder.size() == 0)
              return p;
          build(inorder, postorder, 0, inorder.size() - 1, 0, postorder.size() - 1, p);
          return p;
      }
   };
   ```

     同理可得 105 Construct Binary Tree from Preorder and Inorder Traversal

+  90 Subsets II

        去重：在每一层不选重复元素，到下一层才选，这样就去重了

   ```c++
    class Solution {
    public:
        vector<vector<int>> subsetsWithDup(vector<int>& nums) {
            vector<vector<int>> res;
            vector<int> sub;
            sort(nums.begin(), nums.end());
            btsub(nums, res, sub, 0);
            return res;
        }

        void btsub(vector<int>& nums, vector<vector<int>>& res, vector<int> sub, int n){
            res.push_back(sub);
            vector<int> has;
            for(int i = n; i < nums.size(); i++){
                if(i != n && nums[i] == nums[i - 1]) continue;
                sub.push_back(nums[i]);
                btsub(nums, res, sub, i + 1);
                sub.pop_back();
            }
        }
    };
   ```

+  73 Set Matrix Zeroes

       题目要求空间复杂度为O(1)，所以使用原数组的第一行和第一列来存储需要被替换为0的行列的信息

   ```c++
     class Solution {
       public:
       void setZeroes(vector<vector<int>>& matrix) {
         int row = matrix.size();
         if(row == 0)
           return;
         int col = matrix[0].size(), r0 = 0, c0 = 0;
         for(int i = 0; i < row; i++){
           for(int j = 0; j < col; j++){
             if(matrix[i][j] == 0){
               matrix[0][j] = 0;
               matrix[i][0] = 0;
               if(i == 0)
                 r0 = 1;
               if(j == 0)
                 c0 = 1;
             }
           }
         }
         for(int i = 1; i < row; i++){
           for(int j = 1; j < col; j++){
             if(matrix[0][j] == 0 || matrix[i][0] == 0)
               matrix[i][j] = 0;
           }
         }
         if(r0 == 1){
           for(int j = 0; j < col; j++)
             matrix[0][j] = 0;
         }
         if(c0 == 1){
           for(int i = 0; i < row; i++){
             matrix[i][0] = 0;
           }
         }
       }
     };
   ```

+  33 Search in Rotated Sorted Array

                    同153中用二分查找找出数组的最小点，判断target属于旋转后数组的哪一步分，再用二分查找。

   ```c++
     class Solution {
       public:
       int search(vector<int>& nums, int target) {
         int left = 0, len = nums.size(), right = len - 1, mid, rot;
         for(mid = (left + right) / 2; left < right; mid = (left + right) / 2){
           if(nums[mid] > nums[right])
             left = mid + 1;
           else
             right = mid;
         }
         rot = mid;
         if(target > nums[len - 1]){
           left = 0;
           right = rot - 1;
         }
         else{
           left = rot;
           right = len - 1;
         }
         while(left <= right){
           mid = (left + right) /2 ;
           if(nums[mid] == target)
             return mid;
           if(nums[mid] > target)
             right = mid - 1;
           else
             left = mid + 1;
         }
         return -1;
       }
     };
   ```

+  81 Search in Rotated Sorted Array I / II

   ```c++
     class Solution {
       public:
       int search(vector<int>& nums, int target) {
         for(int left = 0, right = nums.size() - 1, mid = (left + right) / 2; left <= right; mid = (left + right) / 2){
           if(nums[mid] == target) return mid;
           if(nums[left] <= nums[mid] && nums[mid] <= nums[right]){
             if(nums[mid] > target)
               right = mid - 1;
             else
               left = mid + 1;
           }
           else if(nums[left] >= nums[mid] && nums[mid] <= nums[right]){
             if(target < nums[mid] || target > nums[right])
               right = mid - 1;
             else
               left = mid + 1;
           }
           else if(nums[left] <= nums[mid] && nums[mid] >= nums[right]){
             if(target > nums[mid] || target < nums[left])
               left = mid + 1;
             else
               right = mid - 1;
           }
         }
         return -1;
       }
     };
   ```

                   I中的中点所在会有3种情况，II中除了这三种情况外还会有左，中，右三点相等的情况，这时候需要分别对两遍进行二分查找

   ```c++
     class Solution {
       public:
       bool search(vector<int>& nums, int target) {
         return binarysearch(nums, target, 0, nums.size() - 1);
       }
       bool binarysearch(vector<int>& nums, int target, int left, int right){
         for(int mid = (left + right) / 2; left <= right; mid = (left + right) / 2){
           if(nums[mid] == target) return true;
           if(nums[left] == nums[mid] && nums[mid] == nums[right]){
             if(binarysearch(nums, target, left, mid - 1))   return true;
             if(binarysearch(nums,target, mid + 1, right))   return true;
             return false;
           }
           if(nums[left] <= nums[mid] && nums[mid] <= nums[right]){
             if(nums[mid] > target)
               right = mid - 1;
             else
               left = mid + 1;
           }
           else if(nums[left] >= nums[mid] && nums[mid] <= nums[right]){
             if(target < nums[mid] || target > nums[right])
               right = mid - 1;
             else
               left = mid + 1;
           }
           else if(nums[left] <= nums[mid] && nums[mid] >= nums[right]){
             if(target > nums[mid] || target < nums[left])
               left = mid + 1;
             else
               right = mid - 1;
           }
         }
         return false;
       }
     };
   ```

+  229 Majority Element II

   摩尔投票法 Moore Voting，这种方法在之前那道题[Majority Element 求众数](http://www.cnblogs.com/grandyang/p/4233501.html)中也使用了。题目中给了一条很重要的提示，让我们先考虑可能会有多少个众数，经过举了很多例子分析得出，任意一个数组出现次数大于n/3的众数最多有两个，具体的证明我就不会了，我也不是数学专业的。那么有了这个信息，我们使用投票法的核心是找出两个候选众数进行投票，需要两遍遍历，第一遍历找出两个候选众数，第二遍遍历重新投票验证这两个候选众数是否为众数即可，选候选众数方法和前面那篇[Majority Element 求众数](http://www.cnblogs.com/grandyang/p/4233501.html)一样，由于之前那题题目中限定了一定会有众数存在，故而省略了验证候选众数的步骤，这道题却没有这种限定，即满足要求的众数可能不存在，所以要有验证

```c++
class Solution {
public:
    vector<int> majorityElement(vector<int>& nums) {
        vector<int> res;
        int m = 0, n = 0, cm = 0, cn = 0;
        for (auto &a : nums) {
            if (a == m) ++cm;
            else if (a ==n) ++cn;
            else if (cm == 0) m = a, cm = 1;
            else if (cn == 0) n = a, cn = 1;
            else --cm, --cn;
        }
        cm = cn = 0;
        for (auto &a : nums) {
            if (a == m) ++cm;
            else if (a == n) ++cn;
        }
        if (cm > nums.size() / 3) res.push_back(m);
        if (cn > nums.size() / 3) res.push_back(n);
        return res;
    }
};
```

+ 53 Maximum Subarray

  1.用动态规划的方法，就是要找到其转移方程式，也叫动态规划的递推式，动态规划的解法无非是维护两个变量，局部最优和全局最优，我们先来看Maximum SubArray的情况，如果遇到负数，相加之后的值肯定比原值小，但可能比当前值大，也可能小，所以，对于相加的情况，只要能够处理局部最大和全局最大之间的关系即可

  ```c++
  class Solution {
  public:
      int maxSubArray(vector<int>& nums) {
          if(nums.size() == 0)
              return 0;
          int max = nums[0], t = nums[0];
          for(int i = 1; i < nums.size(); i++){
              t = nums[i] + t > nums[i] ? nums[i] + t : nums[i];
              max = t > max ? t : max;
          }
          return max;
      }
  };
  ```

  2.分治法：由于我们知道最大子序列可能存在于A数组的左边，右边，或者一点左边一点右边。 所以我们很容易可以联想到，居然这样我们可以把A数组划分成若干个小的子数组，对子数组求出左边的最大值，和右边的最大值，再求出从中间位置到左边的某个位置的最大值、从中间位置到右边的某个位置的最大值，两个值相加得到包含中点在内的最大值，得到了这四个值之后剩下的我们就可以通过比较得到这个子数组的最大值了。（递归的过程） 

+ 122 Best Time to Buy and Sell Stock II

  每一段单调递增区间的收益累加

+ 34 Search for a Range

  如果我们不寻找那个元素先，而是直接相等的时候也向一个方向继续夹逼，如果向右夹逼，最后就会停在右边界，而向左夹逼则会停在左边界，如此用停下来的两个边界就可以知道结果了，只需要两次二分查找

+ 11 Container With Most Water

  1.首先假设我们找到能取最大容积的纵线为 i , j (假定i<j)，那么得到的最大容积 C = min( ai , aj ) * ( j- i) ；

  2.下面我们看这么一条性质：

  ①: 在 j 的右端没有一条线会比它高！ 假设存在 k |( j<k && ak > aj) ，那么  由 ak> aj，所以 min( ai,aj, ak) =min(ai,aj) ，所以由i, k构成的容器的容积C' = min(ai,aj ) * ( k-i) > C，与C是最值矛盾，所以得证j的后边不会有比它还高的线；

  ②:同理，在i的左边也不会有比它高的线；

  这说明什么呢？如果我们目前得到的候选： 设为 x, y两条线（x< y)，那么能够得到比它更大容积的新的两条边必然在  [x,y]区间内并且 ax' > =ax , ay'>= ay;

  3.所以我们从两头向中间靠拢，同时更新候选值；在收缩区间的时候优先从  x, y中较小的边开始收缩；

  直观的解释是：容积即面积，它受长和高的影响，当长度减小时候，高必须增长才有可能提升面积，所以我们从长度最长时开始递减，然后寻找更高的线来更新候补；
```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int max = 0,l = 0, r = height.size() - 1;
        while(l < r){
            int s = (r - l) * min(height[l], height[r]);
            if( s > max)
                max = s;
            if(height[l] < height[r]){
                int t = height[l];
                while(l < r && height[ ++l ] < t);
            }
            else{
                int t = height[r];
                while(l < r && height[ --r ] < t);
            }
        }
        return max;
    }
};
```
+ 31 Next Permutation

  计算下一个排列的算法：
  设P是1～n的一个全排列:p=p1p2......pn=p1p2......pj-1pjpj+1......pk-1pkpk+1......pn
　　1）从排列的右端开始，找出第一个比右边数字小的数字的序号j（j从左端开始计算），即 j=max{i|pi<pi+1}
　　2）在pj的右边的数字中，找出所有比pj大的数中最小的数字pk，即 k=max{i|pi>pj}（右边的数从右至左是递增的，因此k 是所有大于pj的数字中序号最大者）
　　3）对换pi，pk
　　4）再将pj+1......pk-1pkpk+1......pn倒转得到排列p'=p1p2.....pj-1pjpn.....pk+1pkpk-1.....pj+1，这就是排列  p的下一个排列。
	```c++
  class Solution {
	  public:
    void nextPermutation(vector<int>& nums) {
        if (nums.size() < 2) return;
        int i, k;
        for (i = nums.size() - 2; i >= 0; --i) if (nums[i] < nums[i+1]) break;
        for (k = nums.size() - 1; k > i; --k) if (nums[i] < nums[k]) break;
        if (i >= 0) swap(nums[i], nums[k]);
        reverse(nums.begin() + i + 1, nums.end());
    }
	};
	```
