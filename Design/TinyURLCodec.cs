using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    535. Encode and Decode TinyURL
    https://leetcode.com/problems/encode-and-decode-tinyurl/description/

    */
    public class TinyURLCodec
    {
        /*Approach #1 Using Simple Counter(SC)

        Performance Analysis
            •	The range of URLs that can be decoded is limited by the range of int.
            •	If excessively large number of URLs have to be encoded, after the range of int is exceeded, integer overflow could lead to overwriting the previous URLs' encodings, leading to the performance degradation.
            •	The length of the URL isn't necessarily shorter than the incoming longURL. It is only dependent on the relative order in which the URLs are encoded.
            •	One problem with this method is that it is very easy to predict the next code generated, since the pattern can be detected by generating a few encoded URLs.
        */

        public class TinyURLCodecSC
        {
            private Dictionary<int, string> urlMap = new Dictionary<int, string>();
            private int currentIndex = 0;

            public string Encode(string longUrl)
            {
                urlMap[currentIndex] = longUrl;
                return "http://tinyurl.com/" + currentIndex++;
            }

            public string Decode(string shortUrl)
            {
                return urlMap[int.Parse(shortUrl.Replace("http://tinyurl.com/", ""))];
            }
        }
        /*
        Approach #2 Variable-Length Encoding(VLE)
                
        Performance Analysis
            •	The number of URLs that can be encoded is, again, dependent on the range of int, since, the same count will be generated after overflow of integers.
            •	The length of the encoded URLs isn't necessarily short, but is to some extent dependent on the order in which the incoming longURL's are encountered. For example, the codes generated will have the lengths in the following order: 1(62 times), 2(62 times) and so on.
            •	The performance is quite good, since the same code will be repeated only after the integer overflow limit, which is quite large.
            •	In this case also, the next code generated could be predicted by the use of some calculations.
        */
        public class TinyURLCodecVLE
        {
            private string characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            private Dictionary<string, string> urlMap = new Dictionary<string, string>();
            private int urlCount = 1;

            public string GenerateString()
            {
                int currentCount = urlCount;
                StringBuilder stringBuilder = new StringBuilder();
                while (currentCount > 0)
                {
                    currentCount--;
                    stringBuilder.Append(characters[currentCount % 62]);
                    currentCount /= 62;
                }
                return stringBuilder.ToString();
            }

            public string Encode(string longUrl)
            {
                string key = GenerateString();
                urlMap[key] = longUrl;
                urlCount++;
                return "http://tinyurl.com/" + key;
            }

            public string Decode(string shortUrl)
            {
                return urlMap[shortUrl.Replace("http://tinyurl.com/", "")];
            }
        }

        /*
        Approach #3 Using hashcode (HC)
                
        Performance Analysis	
            •	The number of URLs that can be encoded is limited by the range of int, since hashCode uses integer calculations.
            •	The average length of the encoded URL isn't directly related to the incoming longURL length.
            •	The hashCode() doesn't generate unique codes for different string. This property of getting the same code for two different inputs is called collision. Thus, as the number of encoded URLs increases, the probability of collisions increases, which leads to failure.
            •	The following figure demonstrates the mapping of different objects to the same hashcode and the increasing probability of collisions with increasing number of objects.
        */
        public class TinyURLCodecHC
        {
            Dictionary<int, String> map = new Dictionary<int, string>();

            public String Encode(String longUrl)
            {
                map[longUrl.GetHashCode()] = longUrl;
                return "http://tinyurl.com/" + longUrl.GetHashCode(); ;
            }

            public String Decode(String shortUrl)
            {
                return map[Int32.Parse(shortUrl.Replace("http://tinyurl.com/", ""))];
            }
        }

        /*
        Approach #4 Using random number (RN)
                
        Performance Analysis
            •	The number of URLs that can be encoded is limited by the range of int.
            •	The average length of the codes generated is independent of the longURL's length, since a random integer is used.
            •	The length of the URL isn't necessarily shorter than the incoming longURL. It is only dependent on the relative order in which the URLs are encoded.
            •	Since a random number is used for coding, again, as in the previous case, the number of collisions could increase with the increasing number of input strings, leading to performance degradation.
            •	Determining the encoded URL isn't possible in this scheme, since we make use of random numbers.
        */

        public class TinyURLCodecRN
        {
            private Dictionary<int, string> urlMap = new Dictionary<int, string>();
            private Random randomGenerator = new Random();
            private int uniqueKey;

            public string Encode(string longUrl)
            {
                uniqueKey = randomGenerator.Next(int.MaxValue);
                while (urlMap.ContainsKey(uniqueKey))
                {
                    uniqueKey = randomGenerator.Next(int.MaxValue);
                }
                urlMap[uniqueKey] = longUrl;
                return "http://tinyurl.com/" + uniqueKey;
            }

            public string Decode(string shortUrl)
            {
                return urlMap[int.Parse(shortUrl.Replace("http://tinyurl.com/", ""))];
            }
        }

        /*
        Approach #5 Random fixed-length encoding (RndFLE)
                
        Performance Analysis
            •	The number of URLs that can be encoded is quite large in this case, nearly of the order (10+26∗2)6.
            •	The length of the encoded URLs is fixed to 6 units, which is a significant reduction for very large URLs.
            •	The performance of this scheme is quite good, due to a very less probability of repeated same codes generated.
            •	We can increase the number of encodings possible as well, by increasing the length of the encoded strings. Thus, there exists a tradeoff between the length of the code and the number of encodings possible.
            •	Predicting the encoding isn't possible in this scheme since random numbers are used.

        */
        public class TinyUrlCodecRndFLE
        {
            private string alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            private Dictionary<string, string> urlMap = new Dictionary<string, string>();
            private Random randomGenerator = new Random();
            private string uniqueKey;

            public TinyUrlCodecRndFLE()
            {
                uniqueKey = GenerateRandomKey();
            }

            private string GenerateRandomKey()
            {
                StringBuilder stringBuilder = new StringBuilder();
                for (int i = 0; i < 6; i++)
                {
                    stringBuilder.Append(alphabet[randomGenerator.Next(62)]);
                }
                return stringBuilder.ToString();
            }

            public string Encode(string longUrl)
            {
                while (urlMap.ContainsKey(uniqueKey))
                {
                    uniqueKey = GenerateRandomKey();
                }
                urlMap[uniqueKey] = longUrl;
                return "http://tinyurl.com/" + uniqueKey;
            }

            public string Decode(string shortUrl)
            {
                return urlMap[shortUrl.Replace("http://tinyurl.com/", "")];
            }
        }


    }
    // Your Codec object will be instantiated and called as such:
    // Codec codec = new Codec();
    // codec.decode(codec.encode(url));
}