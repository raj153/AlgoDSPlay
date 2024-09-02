using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2227. Encrypt and Decrypt Strings
    https://leetcode.com/problems/encrypt-and-decrypt-strings/description/

    Complexity
        Encrypter Time O(n) Space O(n)
        encrypt Time O(word1) Space O(word1)
        decrypt Time O(1) Space O(1)

    */
    public class Encrypter
    {
        Dictionary<char, String> enc;
        Dictionary<string, int> count;

        public Encrypter(char[] keys, string[] values, string[] dictionary)
        {
            enc = new  Dictionary<char, string>();
            for (int i = 0; i < keys.Length; ++i)
                enc[keys[i]] =values[i];

            count = new  Dictionary<string, int>();
            foreach (string w in dictionary)
            {
                String e = Encrypt(w);
                count[e] = count.GetValueOrDefault(e, 0) + 1;
            }
        }

        public String Encrypt(String word1)
        {
            StringBuilder res = new StringBuilder();
            for (int i = 0; i < word1.Length; ++i)
                res.Append(enc.GetValueOrDefault(word1[i], "#"));
            return res.ToString();
        }

        public int Decrypt(String word2)
        {
            return count.GetValueOrDefault(word2, 0);
        }

    }
}