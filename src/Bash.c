#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void inputForFileMode(const char* cmd)
{
    // 为清空文件特制
    if (strcmp(cmd, "clear") == 0)
    {
        FILE *output_file = fopen("../BashOutput.txt", "w");
        fclose(output_file);
    }
    else
    {
        char output_buffer[131072];
        FILE *cmd_result = popen(cmd, "r");
        fread(output_buffer, sizeof(char), sizeof(output_buffer), cmd_result);
        FILE *output_file = fopen("../BashOutput.txt", "a");
        system("date >> ../BashOutput.txt");
        fputs(output_buffer, output_file);
        fprintf(output_file, "\n");
        pclose(cmd_result);
        fclose(output_file);
    }
}

char* inputForShellMode(const char *cmd)
{
    int buffer_size = 131072;   // 缓冲区大小128KB
    char* output_buffer = (char*)malloc(sizeof(char) * buffer_size);
    FILE* cmd_result = popen(cmd, "r");
    fread(output_buffer, sizeof(char), buffer_size, cmd_result);
    pclose(cmd_result);
    return output_buffer;
}