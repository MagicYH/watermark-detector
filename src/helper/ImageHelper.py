from PIL import Image

# Image helper, a helper class to preprocess images

class ImageHelper():

    @staticmethod
    def AddWater(sPath, wPath, x1, y1, x2, y2):
        """Add water mark to a image

        Args:
            sPath: source image file path
            wPath: water image file path
            x1, y1, x2, y2: position water will be place into source image, when this value small than 1, it will mean percentage

        Returns:
            image that mark with water
        Raise:
            IOError
            ValueError
        """

        sImg = Image.open(sPath)
        wImg = Image.open(wPath)
        sSize = sImg.size
        waterWidth = x2 - x1
        waterHeight = y2 - y1

        # parameter check
        if waterWidth > sSize[0] or waterHeight > sSize[1]:
            raise ValueError("Invalid area config")
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            raise ValueError("Invalid area config")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid area config")

        if x1 < 1 and x2 < 1 and y1 < 1 and y2 < 1:
            x1 = int(round(sSize[0] * x1))
            x2 = int(round(sSize[0] * x2))
            y1 = int(round(sSize[1] * y1))
            y2 = int(round(sSize[1] * y2))
        
        tImg = wImg.resize((x2 - x1, y2 - y1), Image.ANTIALIAS)
        box = (x1, y1, x2, y2)
        sImg.paste(tImg, box, mask=tImg)
        return sImg

    @staticmethod
    def AddWaterWithImg(sImg, wImg, x1, y1, x2, y2):
        """Add water mark to a image

        Args:
            sImg: source image object
            wImg: water image object
            x1, y1, x2, y2: position water will be place into source image, when this value small than 1, it will mean percentage

        Returns:
            image that mark with water
        Raise:
            IOError
            ValueError
        """
        sSize = sImg.size
        waterWidth = x2 - x1
        waterHeight = y2 - y1

        # parameter check
        if waterWidth > sSize[0] or waterHeight > sSize[1]:
            raise ValueError("Invalid area config")
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            raise ValueError("Invalid area config")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid area config")

        if x1 < 1 and x2 < 1 and y1 < 1 and y2 < 1:
            x1 = int(round(sSize[0] * x1))
            x2 = int(round(sSize[0] * x2))
            y1 = int(round(sSize[1] * y1))
            y2 = int(round(sSize[1] * y2))
        
        cImg = wImg.copy()
        tImg = cImg.resize((x2 - x1, y2 - y1), Image.ANTIALIAS)
        box = (x1, y1, x2, y2)
        # I have to write like this to keep water image transparent
        sImg.paste(tImg, box, mask=tImg)
        return sImg