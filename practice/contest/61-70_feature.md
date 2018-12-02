- 0978_1 :

  ```python
  def fun(featurestr):
      if ("无" in featureStr):
          return 0
      elif ("下腹部疤痕：剖腹产" in featureStr):
          return 1
      elif ("右下腹疤痕：阑尾炎术后" in featureStr):
          return 2
      elif("颈部疤痕：甲状腺良性病变术后" in featureStr or "颈部疤痕：甲状腺良性病变术后" in featureStr):
          return 3 
      else:
          return 4 
  ```

  ​

- 1302_1

  ```python
  def fun(featurestr):
      if ("正常" in featureStr or "未见异常" in featureStr or "正常 ### 正常" in featureStr or "未见异常 ### 未见异常" in featureStr or "未见明显异常" in featureStr):
          return 0
      elif ("双眼慢性结膜炎" in featureStr):
          return 1
      else:
          return 2 
  ```

  ​

- 0405_1

  ```python
  def fun(featurestr):
      if ("未见异常" in featureStr or "未见异常 ### 未见异常" in featureStr or "未闻及" in featureStr ):
          return 0
      elif ("无" in featureStr):
          return 1
      else:
          return 2 
  ```

  ​

- 0975_1

   ```python
  def fun(featurestr):
      if ("未触及" in featureStr or "未触及 ### 未触及" in featureStr ):
          return 0
      elif ("无" in featureStr):
          return 1
      else:
          return 2 
   ```

  ​

- 0976_1

  ```python
  def fun(featurestr):
      if ("无" in featureStr or "无 ### 无" in featureStr ):
          return 0
      else:
          return 1 
  ```

  ​

- 1301_1

  ```python
  def fun(featurestr):
      if ("正常" in featureStr or "未见异常" in featureStr or "正常 ### 正常" in featureStr or "未见异常 ### 未见异常" in featureStr or "正常 ### 正常 ### 正常" in featureStr ):
          return 0
      elif ("双眼睑结膜充血" in featureStr):
          return 1
      else:
          return 2 
  ```

  ​

- 0974_1

  ```Python
  def fun(featurestr):
      if ("无" in featureStr or "无 ### 无" in featureStr ):
          return 0
      else:
          return 1 
  ```

  ​

- 3207_2

  ```python
  def fun(featurestr):
      if ("未见" in featureStr or "未检出" in featureStr ):
          return 0
      elif ("阴性" in featureStr or "-" in featureStr):
          return 1
      else:
          return 2 
  ```

  ​

