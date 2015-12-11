Python 2.7.10 (default, May 23 2015, 09:40:32) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> ================================ RESTART ================================
>>> 
>>> blogs, palabras, datos = readfile('blogdata1.txt')
>>> cluts = hcluster(datos, euclidean)
>>> type(cluts)
<type 'instance'>
>>> printclust(cluts,blogs)
-
  Hispasec @unaaldia
  -
    Mitos y Timos
    -
      El retorno de los charlatanes
      -
        Astrofísica    y    Física
        -
          Experimentos caseros
          -
            Curistoria - Curiosidades y anécdotas históricas
            -
              Genbeta Dev
              -
                Cholonymous: Blog Peruano de Actualidad y Tecnología
                -
                  Xataka Ciencia
                  -
                    Hipertextual
                    -
                      Eureka
                      -
                        Tecnología 7
                        -
                          PC World en Español
                          -
                            PC World Perú
                            -
                              EspacioCiencia.com
                              -
                                Historias de la Historia
                                -
                                  La mentira esta ahi fuera
                                  -
                                    La Ciencia para todos
                                    -
                                      Naukas
                                      -
                                        Tecnología Obsoleta
                                        -
                                          La Ciencia y sus Demonios
                                          -
                                            Círculo Escéptico Argentino
                                            -
                                              FayerWayer
                                              Imagen astronomía diaria - Observatorio
>>> 
