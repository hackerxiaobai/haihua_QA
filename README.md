### 2021海华AI挑战赛·中文阅读理解·技术组
> 我都不会，何苦难为bert

> 欢迎star

[比赛地址](https://www.biendata.xyz/competition/haihua_2021/)
+ 提供一个支持处理超长文本的bert来解QA问题
+ 基于bert4keras库
+ 数据样式
```python
        [
            {
                "ID": 1,
                "Content": "奉和袭美抱疾杜门见寄次韵  陆龟蒙虽失春城醉上期，下帷裁遍未裁诗。因吟郢岸百亩蕙，欲采商崖三秀芝。栖野鹤笼宽使织，施山僧饭别教炊。但医沈约重瞳健，不怕江花不满枝。",
                "Questions": [
                    {
                        "Question": "下列对这首诗的理解和赏析，不正确的一项是",
                        "Choices": [
                            "A．作者写作此诗之时，皮日休正患病居家，闭门谢客，与外界不通音讯。",
                            "B．由于友人患病，原有的约会被暂时搁置，作者游春的诗篇也未能写出。",
                            "C．作者虽然身在书斋从事教学，但心中盼望能走进自然，领略美好春光。",
                            "D．尾联使用了关于沈约的典故，可以由此推测皮日休所患的疾病是目疾。"
                            ],
                        "Answer": "A",
                        "Q_id": "000101"
                    }
                ]
            }
        ]
```

### 解题思路
+ 将content 和 question concat起来，choice 作为下一句，然后做一个二分类
+ predict 的时候，将所有答案的概率值都计算出来，最后选择一个值最大的作为最终结果
+ 看代码吧，不要在意有多粗暴简单，这也只是我花了一个小时写的小demo，希望给大家一个最简单的参考

fea a branch

