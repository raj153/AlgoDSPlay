using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    271. Encode and Decode Strings
    https://leetcode.com/problems/encode-and-decode-strings/description/

    */
    public class StringCodec
    {
        /*
        Approach 1: Non-ASCII Delimiter (NAD)
        Complexity Analysis
        Let n denote the total number of characters across all strings in the input list and k denote the number of strings.
        •	Time Complexity: O(n).
            Both encoding and decoding processes iterate over every character in the input, thus they both have a linear time complexity of O(n).
        •	Space Complexity: O(k).
            We don't count the output as part of the space complexity, but for each word, we are using some space for the delimiter.	

        */
        public class StringCodecNAD
        {
            // Encodes a list of strings to a single string.
            public string Encode(List<string> strings)
            {
                var encodedStringBuilder = new System.Text.StringBuilder();
                // Iterate through the list of strings
                foreach (string str in strings)
                {
                    // Append each string to the StringBuilder followed by the delimiter
                    encodedStringBuilder.Append(str);
                    encodedStringBuilder.Append("π");
                }
                // Return the entire encoded string
                return encodedStringBuilder.ToString();
            }

            // Decodes a single string to a list of strings.
            public List<string> Decode(string encodedString)
            {
                // Split the encoded string at each occurrence of the delimiter
                // Note: We use -1 as the limit parameter to ensure trailing empty strings are included
                string[] decodedStrings = encodedString.Split(new string[] { "π" }, StringSplitOptions.None);
                // Convert the array to a list and return it
                // Note: We remove the last element because it's an empty string resulting from the final delimiter
                return decodedStrings.Take(decodedStrings.Length - 1).ToList();
            }
        }

        /*
        Approach 2: Escape Character and Delimiter (ESD)
        Complexity Analysis
        Let n denote the total number of characters across all strings in the input list and k denote the number of strings.
        •	Time Complexity: O(n).
            Both encoding and decoding processes iterate over every character in the input, thus they both have a linear time complexity of O(n).
        •	Space Complexity: O(k).
            We don't count the output as part of the space complexity, but for each word, we are using escape character and delimiter.	

        */
        public class StringCodecESD
        {
            // Encodes a list of strings to a single string.
            public string Encode(List<string> strings)
            {
                // Initialize a StringBuilder to hold the encoded strings
                StringBuilder encodedString = new StringBuilder();

                // Iterate over each string in the input list
                foreach (string str in strings)
                {
                    // Replace each occurrence of '/' with '//'
                    // This is our way of "escaping" the slash character
                    // Then add our delimiter '/:' to the end
                    encodedString.Append(str.Replace("/", "//")).Append("/:");
                }

                // Return the final encoded string
                return encodedString.ToString();
            }

            // Decodes a single string to a list of strings.
            public List<string> Decode(string encodedString)
            {
                // Initialize a List to hold the decoded strings
                List<string> decodedStrings = new List<string>();

                // Initialize a StringBuilder to hold the current string being built
                StringBuilder currentString = new StringBuilder();

                // Initialize an index 'index' to start of the string
                int index = 0;

                // Iterate while 'index' is less than the length of the encoded string
                while (index < encodedString.Length)
                {
                    // If we encounter the delimiter '/:'
                    if (index + 1 < encodedString.Length && encodedString[index] == '/' && encodedString[index + 1] == ':')
                    {
                        // Add the currentString to the list of decodedStrings
                        decodedStrings.Add(currentString.ToString());

                        // Clear currentString for the next string
                        currentString = new StringBuilder();

                        // Move the index 2 steps forward to skip the delimiter
                        index += 2;
                    }
                    // If we encounter an escaped slash '//'
                    else if (index + 1 < encodedString.Length && encodedString[index] == '/' && encodedString[index + 1] == '/')
                    {
                        // Add a single slash to the currentString
                        currentString.Append('/');

                        // Move the index 2 steps forward to skip the escaped slash
                        index += 2;
                    }
                    // Otherwise, just add the character to currentString and move the index 1 step forward.
                    else
                    {
                        currentString.Append(encodedString[index]);
                        index++;
                    }
                }

                // Return the list of decoded strings
                return decodedStrings;
            }
            /*
            Approach 3: Chunked Transfer Encoding (CTE)
            Complexity Analysis
            Let n denote the total number of characters across all strings in the input list and k denote the number of strings.
            •	Time Complexity: O(n).
                We are iterating through each string once.
            •	Space Complexity: O(k).
                We don't count the output as part of the space complexity, but for each word, we are using some space for the length and delimiter.	
            */
            public class StringCodecCTE
            {
                public string Encode(List<string> strings)
                {
                    // Initialize a StringBuilder to hold the encoded string.
                    StringBuilder encodedString = new StringBuilder();
                    foreach (string str in strings)
                    {
                        // Append the length, the delimiter, and the string itself.
                        encodedString.Append(str.Length).Append("/:").Append(str);
                    }
                    return encodedString.ToString();
                }

                public List<string> Decode(string encodedString)
                {
                    // Initialize a list to hold the decoded strings.
                    List<string> decodedStrings = new List<string>();
                    int index = 0;
                    while (index < encodedString.Length)
                    {
                        // Find the delimiter.
                        int delimiterIndex = encodedString.IndexOf("/:", index);
                        // Get the length, which is before the delimiter.
                        int length = int.Parse(encodedString.Substring(index, delimiterIndex - index));
                        // Get the string, which is of 'length' length after the delimiter.
                        string str = encodedString.Substring(delimiterIndex + 2, length);
                        // Add the string to the list.
                        decodedStrings.Add(str);
                        // Move the index to the start of the next length.
                        index = delimiterIndex + 2 + length;
                    }
                    return decodedStrings;
                }
            }

        }

    }
}