Python 2.7.10 (default, May 23 2015, 09:40:32) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> ================================ RESTART ================================
>>> 
>>> blogs, palabras, datos = readfile('blogdata1.txt')
>>> cluts = hcluster(datos, manjathan)
>>> type(cluts)
<type 'instance'>
>>> printclust(cluts,blogs)
-
  Mitos y Timos
  -
    -
      -
        -
          Cholonymous: Blog Peruano de Actualidad y Tecnología
          Xataka Ciencia
        -
          Genbeta Dev
          -
            Astrofísica    y    Física
            -
              Curistoria - Curiosidades y anécdotas históricas
              Experimentos caseros
      -
        Hipertextual
        -
          -
            PC World Perú
            Tecnología 7
          -
            -
              Imagen astronomía diaria - Observatorio
              -
                Naukas
                Historias de la Historia
            -
              -
                Eureka
                -
                  La mentira esta ahi fuera
                  PC World en Español
              -
                EspacioCiencia.com
                -
                  -
                    Tecnología Obsoleta
                    FayerWayer
                  -
                    La Ciencia y sus Demonios
                    -
                      Círculo Escéptico Argentino
                      La Ciencia para todos
    -
      El retorno de los charlatanes
      Hispasec @unaaldia
>>> 
