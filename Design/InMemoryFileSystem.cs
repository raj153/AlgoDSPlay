using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reflection.Metadata;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{

    /*
    588. Design In-Memory File System
    https://leetcode.com/problems/design-in-memory-file-system/

    //Approach #1 Using separate Directory and File List
    */
    public class InMemoryFileSystem
    {

        class Dir
        {
            public Dictionary<string, Dir> Dirs = new Dictionary<string, Dir>();
            public Dictionary<string, string> Files = new Dictionary<string, string>();
        }
        Dir root;

        public InMemoryFileSystem()
        {
            root = new Dir();

        }
        /*
        The time complexity of executing an ls command is O(m+n+klog(k)). Here, m refers to the length of the input string. 
        We need to scan the input string once to split it and determine the various levels. 
        n refers to the depth of the last directory level in the given input for ls. 
        This factor is taken because we need to enter n levels of the tree structure to reach the last level. k refers to the number of entries(files+subdirectories) in the last level directory(in the current input). 
        We need to sort these names giving a factor of klog(k).
        */
        public List<string> Ls(string path)
        {
            Dir dir = root;

            List<string> files = new List<string>();
            if (!path.Equals("/"))
            {
                string[] dirArr = path.Split("/");
                for (int i = 0; i < dirArr.Length - 1; i++)
                {
                    dir = dir.Dirs[dirArr[i]];
                }
                if (dir.Files.ContainsKey(dirArr[dirArr.Length - 1]))
                {
                    files.Add(dirArr[dirArr.Length - 1]);
                    return files;
                }
                else
                {
                    dir = dir.Dirs[dirArr[dirArr.Length - 1]];
                }
            }
            files.AddRange(new List<string>(dir.Dirs.Keys));
            files.AddRange(new List<string>(dir.Files.Keys));
            files.Sort();
            return files;

        }
        /*
        The time complexity of executing an mkdir command is O(m+n). Here, m refers to the length of the input string. We need to scan the input string once to split it and determine the various levels. 
                n refers to the depth of the last directory level in the mkdir input. This factor is taken because we need to enter n levels of the tree structure to reach the last level.
        */
        public void Mkdir(string path)
        {
            Dir curr = root;
            String[] dirArr = path.Split("/");
            foreach (string dir in dirArr)
            {
                if (!curr.Dirs.ContainsKey(dir))
                    curr.Dirs.Add(dir, new Dir());

                curr = curr.Dirs[dir];
            }
        }
        /*
        The time complexity of addContentToFile is O(m+n). 
        Here, m refers to the length of the input string. We need to scan the input string once to split it and determine the various levels.
        n refers to the depth of the file name in the current input. This factor is taken because we need to enter n levels of the tree structure to reach the level where the files's contents need to be added/read from.
        */
        public void AddContentToFile(string filePath, string content)
        {
            Dir curr = root;
            string[] pathArr = filePath.Split("/");
            for (int i = 0; i < pathArr.Length - 1; i++)
            {
                if (string.IsNullOrEmpty(pathArr[i])) continue;
                curr = curr.Dirs[pathArr[i]];
            }
            string fileName = pathArr[pathArr.Length - 1];
            if (!curr.Files.ContainsKey(fileName))
            {
                curr.Files.Add(fileName, "");
                
            }
            curr.Files[fileName] += content;
            
        }

        /*
        The time complexity of addContentToFile is O(m+n). 
        Here, m refers to the length of the input string. We need to scan the input string once to split it and determine the various levels.
        n refers to the depth of the file name in the current input. This factor is taken because we need to enter n levels of the tree structure to reach the level where the files's contents need to be added/read from.
        */

        public string ReadContentFromFile(string filePath)
        {            
            Dir curr = root;

            string[] pathArr = filePath.Split("/");
            for (int i = 0; i < pathArr.Length - 1; i++)
            {
                if (string.IsNullOrEmpty(pathArr[i])) continue;
                curr = curr.Dirs[pathArr[i]];
            }

            string fileName = pathArr[pathArr.Length - 1];

            return curr.Files[fileName];
        }



        //Approach #2 Using unified Directory and File List[
        public class InMemoryFileSystemOptimal
        {

            class File
            {
                public bool IsFile = false;
                public string Content = "";
                public Dictionary<string, File> Files = new Dictionary<string, File>();
            }
            File root;

            public InMemoryFileSystemOptimal()
            {
                root = new File();

            }
            /*
            The time complexity of executing an ls command is O(m+n+klog(k)). Here, m refers to the length of the input string. 
            We need to scan the input string once to split it and determine the various levels. 
            n refers to the depth of the last directory level in the given input for ls. 
            This factor is taken because we need to enter n levels of the tree structure to reach the last level. k refers to the number of entries(files+subdirectories) in the last level directory(in the current input). 
            We need to sort these names giving a factor of klog(k).
            */
            public List<string> Ls(string path)
            {
                File file = root;

                List<string> files = new List<string>();
                if (!path.Equals("/"))
                {
                    string[] pathArr = path.Split("/");
                    for (int i = 0; i < pathArr.Length -1; i++)
                    {
                        file = file.Files[pathArr[i]];
                    }
                    if (file.IsFile)
                    {
                        files.Add(pathArr[pathArr.Length - 1]);
                        return files;
                    }
                    else
                    {
                        file = file.Files[pathArr[pathArr.Length - 1]];
                    }
                }
                files.AddRange(new List<string>(file.Files.Keys));
                files.Sort();
                return files;

            }
            /*
            The time complexity of executing an mkdir command is O(m+n). Here, m refers to the length of the input string. We need to scan the input string once to split it and determine the various levels. 
                    n refers to the depth of the last directory level in the mkdir input. This factor is taken because we need to enter n levels of the tree structure to reach the last level.
            */
            public void Mkdir(string path)
            {
                File file = root;
                String[] pathArr = path.Split("/");
                for (int i = 1; i < pathArr.Length; i++)
                {
                    if (!file.Files.ContainsKey(pathArr[i]))
                        file.Files.Add(pathArr[i], new File());

                    file = file.Files[pathArr[i]];
                }
            }
            /*
            The time complexity of addContentToFile is O(m+n). 
            Here, m refers to the length of the input string. We need to scan the input string once to split it and determine the various levels.
            n refers to the depth of the file name in the current input. This factor is taken because we need to enter n levels of the tree structure to reach the level where the files's contents need to be added/read from.
            */
            public void AddContentToFile(string filePath, string content)
            {
                File file = root;
                string[] pathArr = filePath.Split("/");
                for (int i = 1; i < pathArr.Length - 1; i++)
                {
                    file = file.Files[pathArr[i]];
                }
                if (!file.Files.ContainsKey(pathArr[pathArr.Length - 1]))
                    file.Files[pathArr[pathArr.Length - 1]] = new File();

                file = file.Files[pathArr[pathArr.Length - 1]];
                file.IsFile = true;
                file.Content += content;


            }

            /*
            The time complexity of  readContentFromFile is O(m+n). 
            Here, m refers to the length of the input string. We need to scan the input string once to split it and determine the various levels.
            n refers to the depth of the file name in the current input. This factor is taken because we need to enter n levels of the tree structure to reach the level where the files's contents need to be added/read from.
            */
            public String ReadContentFromFile(String filePath)
            {
                File t = root;
                String[] d = filePath.Split("/");
                for (int i = 1; i < d.Length - 1; i++)
                {
                    t = t.Files[d[i]];
                }
                return t.Files[d[d.Length - 1]].Content;
            }
        }
    }

    /**
 * Your FileSystem object will be instantiated and called as such:
 * FileSystem obj = new FileSystem();
 * IList<string> param_1 = obj.Ls(path);
 * obj.Mkdir(path);
 * obj.AddContentToFile(filePath,content);
 * string param_4 = obj.ReadContentFromFile(filePath);
 */
}