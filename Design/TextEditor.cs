using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2296. Design a Text Editor	
    https://leetcode.com/problems/design-a-text-editor/description/

    Time Complexity of each operation:
        addText(string text) : O(n) {where n == length of text}
        deleteText(int k) : O(k)
        cursorLeft(int k) : O(k)
        cursorRight(int k) : O(k)
    Space Complexity: O(m) where m is total text

    */
    public class TextEditor
    {
        //Approach 1 : Using Two Stocks
        //Always maintain the left part of string in left stack and right part of the string in right stack which are divided by the cursor
        public class TextEditorTwoStack
        {
            private Stack<char> leftStack = new Stack<char>();
            private Stack<char> rightStack = new Stack<char>();

            public void AddText(string text)
            {
                for (int i = 0; i < text.Length; i++)
                {
                    leftStack.Push(text[i]);
                }
            }

            public int DeleteText(int k)
            {
                int countDeleted = 0;
                while (leftStack.Count > 0 && k-- > 0)
                {
                    leftStack.Pop();
                    countDeleted++;
                }
                return countDeleted;
            }

            public string CursorLeft(int k)
            {
                while (leftStack.Count > 0 && k-- > 0)
                {
                    rightStack.Push(leftStack.Pop());
                }
                return GetLeftString();
            }

            public string CursorRight(int k)
            {
                while (rightStack.Count > 0 && k-- > 0)
                {
                    leftStack.Push(rightStack.Pop());
                }
                return GetLeftString();
            }

            private string GetLeftString()
            {
                int count = 10;
                StringBuilder stringBuilder = new StringBuilder();
                while (leftStack.Count > 0 && count-- > 0)
                {
                    stringBuilder.Append(leftStack.Pop());
                }

                for (int i = stringBuilder.Length - 1; i >= 0; i--)
                {
                    leftStack.Push(stringBuilder[i]);
                }
                return stringBuilder.ToString();
            }
        }

        //Approach 2 : Using StringBuilder(SB)
        public class TextEditorSB
        {
            StringBuilder sb;
            int cursorPosition = 0;

            public TextEditorSB()
            {
                sb = new();
            }

            public void AddText(string text)
            {
                sb.Insert(cursorPosition, text);
                cursorPosition += text.Length;
            }

            public int DeleteText(int k)
            {
                int deletedAmount = k;

                if (cursorPosition < k)
                {
                    deletedAmount = cursorPosition;
                    cursorPosition = 0;
                }
                else
                {
                    cursorPosition -= k;
                }

                sb.Remove(cursorPosition, deletedAmount);

                return deletedAmount;
            }

            public string CursorLeft(int k)
            {
                cursorPosition = Math.Max(0, cursorPosition - k);

                if (cursorPosition < 10)
                {
                    return sb.ToString(0, cursorPosition);
                }

                return sb.ToString(cursorPosition - 10, 10);
            }

            public string CursorRight(int k)
            {
                cursorPosition = Math.Min(sb.Length, cursorPosition + k);

                if (cursorPosition < 10)
                {
                    return sb.ToString(0, cursorPosition);
                }

                return sb.ToString(cursorPosition - 10, 10);
            }
        }
    }

}