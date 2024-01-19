# 简介

计算机相关学习代码

# git 使用说明

## git 在不同平台协作使得回车转化

git自带的有crlf和lf的转化.

管他妈的!, 我只要知道, 在linux上面使用dos2unix命令可以转化

```
dos2unix的一些命令
0.   -k日期不变
1.   -n 源文件不变转化到另一个文件
```

## autocrlf的工作原理

0. true
   *.   push: ??? -> lf
   *.   pull : ??? -> crlf
1. false
   *.   push: none
   *.   pull: none
2. input
   *.   push: ??? -> lf
   *.   pull: none
