import os,sys,pygame,random,math,copy,time
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime


from PIL import Image,ImageDraw
import string

## important: file size: 400 * 200


# 80 * 20
height = 400
width = 200
dx = 5
dy = 10
row_num = height // dx
col_num = width // dy


def rgbString(red, green, blue):
    return "#%02x%02x%02x" % (red, green, blue)

def find_first (x1,y1,x2,y2):
        k = (y2-y1)/(x2-x1)
        b = y1 - x1*k
        return (k,b)



def convert_num_to_rgb_10_1 (n):
    # n should be a float
    # n should be between -8000 and 8000
    m = (n+10000) * 10
    r = int(m // 10**4)
    g = int(m // 10**2 - r * 100)
    b = int(m % 100)
    return (r,g,b)

def convert_num_to_rgb_10_3 (n):
    # n should be a float
    # n should be between -499 and 499
    m = (n+500) * 1000
    r = int(m // 10**4)
    g = int(m // 10**2 - r * 100)
    b = int(m % 100)
    return (r,g,b)

def convert_initial_list(A):
    # A is a 80*20 2-D list
    k = list()
    for i in range(len(A)):
        l = list()
        for j in range(len(A[i])):
            if(0 <= j <= 9):
                l.append(convert_num_to_rgb_10_3(A[i][j]))
            else:
                l.append(convert_num_to_rgb_10_1(A[i][j]))
        k.append(l)
    return k

def get_pixel(A,x,y):
    # x: row number
    # y: col number
    # assume the pic is 400 * 360
    if ( x >= height or y >= width):
        return (0,0,0)
    x0 = (x // dx) 
    y0 = (y // dy)
    
    return A[x0][y0]

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))


def draw(RGBs,i):
        height = 400
        width = 200
        rowHeight = 5
        colWidth = 10
        #PILdraw
        #[(min_R,range_R),(min_G,range_G),(min_B,range_B)] = findRange(RGBs,height,width,rowHeight,colWidth)
        image1 = Image.new("RGB", (width, height),'white')
        draw = ImageDraw.Draw(image1)
        #draw
        S = convert_initial_list(RGBs)
        for row in range(0,height):
                print(row)
                for col in range(0,width):
                        x1 = row
                        y1 = col
                        x2 = row + 1
                        y2 = col + 1
                        #(R,G,B) = get_pixel(RGBs,x1,y1)
                        #R = int((R - min_R)*255/range_R)
                        #G = int((G - min_G)*255/range_G)
                        #B = int((B - min_B)*255/range_B)
                        #color = (R,G,B)
                        #PILdraw
                        color = get_pixel(S,x1,y1)
                        draw.rectangle([(y1,x1),(y2,x2)],fill=color)
        name = '%d.jpeg'%i  #need coordination
        #PIL save
        image1.save(name)

def combineData(alphaList,omegaList):
    result = list()
    for row in range(row_num):
        l = list()
        for col in range(col_num):
            temp = col_num//2
            if(col<temp):
                l.append(alphaList[row][col])
            else:
                l.append(omegaList[row][col-temp])
        result.append(l)
    return result

def getListFromFile(oname,aname):
    omegaList = list()
    alphaList = list()
    omegaL = list()
    alphaL = list()
    count = 0
    with open(oname) as omegaFile:
        lines = omegaFile.read().splitlines()
        for line in lines:
            l = list()
            line = line[1:-1]
            for omega in line.split(','):
                l.append(omega)
            omegaList.append(l)
            count += 1
            if(count == 80):
                break
    count = 0
    with open(aname) as alphaFile:
        lines = alphaFile.read().splitlines()
        for line in lines:
            l = list()
            line = line[1:-1]
            for alpha in line.split(','):
                l.append(alpha)
            alphaList.append(l)
            count += 1
            if(count == 80):
                break
    for k in range(len(omegaList)):
        ol = list()
        al = list()
        for j in range(len(omegaList[0])):
            ol.append(float(omegaList[k][j]))
            al.append(float(alphaList[k][j]))
        omegaL.append(ol)
        alphaL.append(al)
    return combineData(omegaL,alphaL)

def drawFromFiles(startIndex,endIndex):
    for i in range(startIndex,endIndex,1):
        oname = "omega%d.txt"%i
        aname = "alpha%d.txt"%i
        print("generating %d picture"%i)
        draw(getListFromFile(oname,aname),i)

def almostEqual(x1,x2):
    return abs(x1-x2)<=0.0001


class Cam(object):
    def __init__(self,pos=(0,0,0),rot=(0,0)):
        self.pos=list(pos)
        self.rot=list(rot)
    def update(self,key,incre):
        s=incre
        if key[pygame.K_q]:self.pos[1]+=s
        if key[pygame.K_e]:self.pos[1]-=s

        if key[pygame.K_w]:self.pos[2]+=s
        if key[pygame.K_s]:self.pos[2]-=s

        if key[pygame.K_a]:self.pos[0]+=s
        if key[pygame.K_d]:self.pos[0]-=s

        if key[pygame.K_UP]:self.rot[0]+=0.01
        if key[pygame.K_DOWN]:self.rot[0]-=0.01
        if key[pygame.K_LEFT]:self.rot[1]+=0.01
        if key[pygame.K_RIGHT]:self.rot[1]-=0.01

class Skeleton(object):
    @staticmethod
    def rotate2d(pos,rad):
        x,y=pos 
        s,c=math.sin(rad),math.cos(rad)
        return (x*c-y*s,y*c+x*s)

    def __init__(self,verts,edges,init_dist):
        self.verts=verts
        self.edges=edges
        self.cam=Cam((0,0,-init_dist))
        self.renderedEdge=[]
        return
    
    def update(self,key,verts=0):
        if verts!=0 : self.verts=verts    
        self.cam.update(key,0.1)
        return
    
    def render(self,cx,cy):
        self.renderedEdge=[]
        for edge in self.edges:
                points=[]
                for x,y,z in (self.verts[edge[0]],self.verts[edge[1]]):
                    x-=self.cam.pos[0]
                    y-=self.cam.pos[1]
                    z-=self.cam.pos[2]
                    x,z=Skeleton.rotate2d((x,z),self.cam.rot[1])
                    y,z=Skeleton.rotate2d((y,z),self.cam.rot[0])
                    f=200/z
                    x,y=x*f,y*f
                    points+=[(cx+int(x),cy-int(y))]
                self.renderedEdge.append(points)
        return

    def draw(self,screen):
        if self.renderedEdge==[]: pass
        #print(self.renderedEdge)
        for points in self.renderedEdge:
            pygame.draw.line(screen,(200,200,200),points[0],points[1],1)
        return



class KinectControl(object):
    '''
    The following code in this class uses the
    PyKinectV2 module developed by microsoft 
    and uses various methods and existing code 
    from the PyKinect module
    '''
    @staticmethod
    def crossProduct(pose1,pose2):
        (x1,y1,z1)=pose1
        (x2,y2,z2)=pose2
        return (y1*z1-y2*z1,-(x1*z2-x2*z1),x1*y2-y1*x2)
    
    
    @staticmethod
    def vectorAdd(pose1,pose2):
        return (pose1[0]+pose2[0],pose1[1]+pose2[1],pose1[2]+pose2[2])

    @staticmethod
    def vectorSubtract(pose1,pose2):
        return (pose1[0]-pose2[0],pose1[1]-pose2[1],pose1[2]-pose2[2])


    @staticmethod
    def getPosition(joint):
        x=joint.Position.x
        y=joint.Position.y
        z=joint.Position.z
        return (x,y,z)

    @staticmethod
    def reflect(pos,refpos):
        dx=refpos(0)-pos(0)
        dy=refpos(1)-pos(1)
        dz=refpos(2)-pos(2)
        return (refpos(0)+dx,refpos(1)+dy,refpos(2)+dz)

    @staticmethod
    def getAngle(pos1,pos2,pos3):
        x1,y1,z1=pos1[0]-pos2[0],pos1[1]-pos2[1],pos1[2]-pos2[2]
        x2,y2,z2=pos3[0]-pos2[0],pos3[1]-pos2[1],pos3[2]-pos2[2]
        d1=((x1)**2+(y1)**2+(z1)**2)**0.5
        d2=((x2)**2+(y2)**2+(z2)**2)**0.5
        dot=x1*x2+y1*y2+z1*z2
        angle=math.acos(dot/(d1*d2+0.00001))
        return math.floor(angle*(180/math.pi))
        
    
    
    def __init__(self,joints):
        self.LShoulder=joints[PyKinectV2.JointType_ShoulderLeft]
        self.RShoulder=joints[PyKinectV2.JointType_ShoulderRight]
        self.LElbow=joints[PyKinectV2.JointType_ElbowLeft]
        self.RElbow=joints[PyKinectV2.JointType_ElbowRight]
        self.LHip=joints[PyKinectV2.JointType_HipLeft]
        self.RHip=joints[PyKinectV2.JointType_HipRight] 
        self.LKnee=joints[PyKinectV2.JointType_KneeLeft]
        self.RKnee=joints[PyKinectV2.JointType_KneeRight] 
        self.LWrist=joints[PyKinectV2.JointType_WristLeft]
        self.RWrist=joints[PyKinectV2.JointType_WristRight]
        self.LAnkle=joints[PyKinectV2.JointType_AnkleLeft]
        self.RAnkle=joints[PyKinectV2.JointType_AnkleRight]
        self.vert=[self.getPosition(self.LShoulder),self.getPosition(self.RShoulder),self.getPosition(self.LElbow),
                                                 self.getPosition(self.RElbow),self.getPosition(self.LHip),self.getPosition(self.RHip),
                                                 self.getPosition(self.LKnee),self.getPosition(self.RKnee),self.getPosition(self.LWrist),
                                                 self.getPosition(self.RWrist),self.getPosition(self.LAnkle),self.getPosition(self.RAnkle)]
        self.bearing=self.vectorAdd(self.crossProduct(self.vectorSubtract(self.vert[4],self.vert[1]),self.vectorSubtract(self.vert[0],self.vert[1])),
                                             self.crossProduct(self.vectorSubtract(self.vert[1],self.vert[0]),self.vectorSubtract(self.vert[5],self.vert[0])))
       
        self.newAngleList=[self.getAngle(self.vectorAdd(self.vert[1],self.bearing),self.vert[1],self.vert[3]),self.getAngle(self.vert[1],self.vert[3],self.vert[9]),
                           self.getAngle(self.vectorAdd(self.vert[5],self.bearing),self.vert[5],self.vert[7]),self.getAngle(self.vert[5],self.vert[7],self.vert[11]),
                           self.getAngle(self.vectorAdd(self.vert[0],self.bearing),self.vert[0],self.vert[2]),self.getAngle(self.vert[0],self.vert[2],self.vert[8]),
                           self.getAngle(self.vectorAdd(self.vert[4],self.bearing),self.vert[4],self.vert[6]),self.getAngle(self.vert[4],self.vert[6],self.vert[10]),
                           self.getAngle(self.vert[0],self.vert[1],(self.vert[0][0],self.vert[0][1],self.vert[1][2])),
                           self.getAngle(self.vert[4],self.vert[5],(self.vert[4][0],self.vert[4][1],self.vert[5][2]))]
        self.oldAngleList=self.newAngleList
        self.newOmegaList=self.oldOmegaList=[0,0,0,0,0,0,0,0,0,0]
        self.alphaList=[0,0,0,0,0,0,0,0,0,0]
        self.was=self.now=time.time()
        self.bearingAngle=self.getAngle(self.bearing,(0,0,0),(1,0,0))
        self.update_iteration=1
        self.update=False

    def update_joints(self,joints):
        self.update_iteration+=1
        self.LShoulder=joints[PyKinectV2.JointType_ShoulderLeft]
        self.RShoulder=joints[PyKinectV2.JointType_ShoulderRight]
        self.LElbow=joints[PyKinectV2.JointType_ElbowLeft]
        self.RElbow=joints[PyKinectV2.JointType_ElbowRight]
        self.LHip=joints[PyKinectV2.JointType_HipLeft]
        self.RHip=joints[PyKinectV2.JointType_HipRight] 
        self.LKnee=joints[PyKinectV2.JointType_KneeLeft]
        self.RKnee=joints[PyKinectV2.JointType_KneeRight] 
        self.LWrist=joints[PyKinectV2.JointType_WristLeft]
        self.RWrist=joints[PyKinectV2.JointType_WristRight]
        self.LAnkle=joints[PyKinectV2.JointType_AnkleLeft]
        self.RAnkle=joints[PyKinectV2.JointType_AnkleRight]
        self.vert=[self.getPosition(self.LShoulder),self.getPosition(self.RShoulder),self.getPosition(self.LElbow),
                                                 self.getPosition(self.RElbow),self.getPosition(self.LHip),self.getPosition(self.RHip),
                                                 self.getPosition(self.LKnee),self.getPosition(self.RKnee),self.getPosition(self.LWrist),
                                                 self.getPosition(self.RWrist),self.getPosition(self.LAnkle),self.getPosition(self.RAnkle)]
        self.bearing=self.vectorAdd(self.crossProduct(self.vectorSubtract(self.vert[4],self.vert[1]),self.vectorSubtract(self.vert[0],self.vert[1])),
                                             self.crossProduct(self.vectorSubtract(self.vert[1],self.vert[0]),self.vectorSubtract(self.vert[5],self.vert[0])))
        
        self.newAngleList=[self.newAngleList[0]+self.getAngle(self.vectorAdd(self.vert[1],self.bearing),self.vert[1],self.vert[3]),self.newAngleList[1]+self.getAngle(self.vert[1],self.vert[3],self.vert[9]),
                           self.newAngleList[2]+self.getAngle(self.vectorAdd(self.vert[5],self.bearing),self.vert[5],self.vert[7]),self.newAngleList[3]+self.getAngle(self.vert[5],self.vert[7],self.vert[11]),
                           self.newAngleList[4]+self.getAngle(self.vectorAdd(self.vert[0],self.bearing),self.vert[0],self.vert[2]),self.newAngleList[5]+self.getAngle(self.vert[0],self.vert[2],self.vert[8]),
                           self.newAngleList[6]+self.getAngle(self.vectorAdd(self.vert[4],self.bearing),self.vert[4],self.vert[6]),self.newAngleList[7]+self.getAngle(self.vert[4],self.vert[6],self.vert[10]),
                           self.newAngleList[8]+self.getAngle(self.vert[0],self.vert[1],(self.vert[0][0],self.vert[0][1],self.vert[1][2])),
                           self.newAngleList[9]+self.getAngle(self.vert[4],self.vert[5],(self.vert[4][0],self.vert[4][1],self.vert[5][2]))]

    def updateJoints(self,joints):
        self.now=time.time()
        dt=self.now-self.was
        if dt<0.025: 
            self.update_joints(joints)
        else:
            self.newAngleList=list(map(lambda x:x//self.update_iteration,self.newAngleList))
            #self.newAngleList=list(map(lambda x:x/100,self.newAngleList))
            #print(self.newAngleList)
            self.oldOmegaList = self.newOmegaList
       
        
        
            #dt+=(time.time()-self.now)
            self.newOmegaList=[(self.newAngleList[0]-self.oldAngleList[0])/dt,(self.newAngleList[1]-self.oldAngleList[1])/dt,
                            (self.newAngleList[2]-self.oldAngleList[2])/dt,(self.newAngleList[3]-self.oldAngleList[3])/dt,
                            (self.newAngleList[4]-self.oldAngleList[4])/dt,(self.newAngleList[5]-self.oldAngleList[5])/dt,
                            (self.newAngleList[6]-self.oldAngleList[6])/dt,(self.newAngleList[7]-self.oldAngleList[7])/dt,
                            (self.newAngleList[8]-self.oldAngleList[8])/dt,(self.newAngleList[9]-self.oldAngleList[9])/dt]
            self.alphaList=[(self.newOmegaList[0]-self.oldOmegaList[0])/dt,(self.newOmegaList[1]-self.oldOmegaList[1])/dt,
                            (self.newOmegaList[2]-self.oldOmegaList[2])/dt,(self.newOmegaList[3]-self.oldOmegaList[3])/dt,
                            (self.newOmegaList[4]-self.oldOmegaList[4])/dt,(self.newOmegaList[5]-self.oldOmegaList[5])/dt,
                            (self.newOmegaList[6]-self.oldOmegaList[6])/dt,(self.newOmegaList[7]-self.oldOmegaList[7])/dt,
                            (self.newOmegaList[8]-self.oldOmegaList[8])/dt,(self.newOmegaList[9]-self.oldOmegaList[9])/dt]
        
            self.was=time.time()
            self.oldAngleList=self.newAngleList
            self.newAngleList=[0,0,0,0,0,0,0,0,0,0]
            #self.bearingAngle=self.getAngle(self.bearing,(0,0,0),(1,0,0))
            self.update_iteration=1
            self.update=True
        #print(self.newOmegaList)
   
class KinectDataProcessor(object):
    
    def __init__(self,joints):
        self.kinect=KinectControl(joints)
        self.o_data=[]
        self.a_data=[]
        self.n=0
        self.newData=False
        self.dataLen=0
        

    def getData(self):
        if not self.kinect.update: return
        if self.dataLen>=80: self.dataLen=0;self.newData=True;return
        self.o_data.append(list(map(lambda x:x*499/800,self.kinect.newOmegaList)))
        self.a_data.append(list(map(lambda x:x*4999/10000,self.kinect.alphaList)))
        self.dataLen+=1
        self.kinect.update=False

    def updateData(self,joints):
        self.kinect.updateJoints(joints)

    def ready4Img(self):
        if not self.newData: return
        name1 = "alpha%d.txt" %self.n
        f=open(name1,"w")
        f.write("\n".join(map(lambda x: str(x), self.a_data)))
        f.close()
        name2="omega%d.txt" %self.n
        g=open(name2,"w")
        g.write("\n".join(map(lambda x: str(x), self.o_data)))
        g.close()
        
        self.a_data=[]
        self.o_data=[]
        self.newData=False
        self.n+=1
        print(0)
        



    

    
    

    

    
    
    
    #def updateKeys(self):

def processData(o_data,a_data):
    new_o = list()
    new_a = list()
    
    l=len(o_data) # 2
    
    print(l)
    
    m=80 #len(o_data[0]) # 3
    n=10 # len(o_data[0][0]) # 3
    r1_o = list()
    r1_a = list()
    for i in range(0,l):
        for j in range(m):
            r2_o = list()
            r2_a = list()
            for k in range(n):
                o = (o_data[i][j][k])*499/800
                a = a_data[i][j][k]
                r2_a.append(a)
                r2_o.append(o)
            r1_o.append(r2_o)
            r1_a.append(r2_a)
    return (r1_o,r1_a)


            
            


       

'''
def validate(n):
    pygame.init()
    screen=pygame.display.set_mode((800,600))
    generate(2*n+1,screen)
    #drawFromFiles(0,n)
    screen.fill((255,255,255))
    displayImgs(screen)
    pygame.quit()
    sys.exit()
    return 
'''



def generate(duration,screen):
    w,h=800,600; cx,cy=400,300

    clock=pygame.time.Clock()
    Kinect=PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body)
    key=pygame.key.get_pressed()
    result=[]
    control=None
    while True:
         if Kinect.has_new_body_frame():
            bodies=Kinect.get_last_body_frame()
            if bodies is not None:
                for i in range(0,Kinect.max_body_count):
                    body=bodies.bodies[i]
                    if not body.is_tracked:
                        continue 
                    joints=body.joints
                    if control==None:
                        control=KinectDataProcessor(joints)
                        verts=control.kinect.vert
                        edges=(0,1),(0,2),(1,3),(4,5),(4,6),(7,5),(8,2),(9,3),(10,6),(11,7),(0,5),(1,4)
                        BODY=Skeleton(verts,edges,0.5)
                        key=pygame.key.get_pressed()
                        startTime=time.time()
                        print(1000)
                    elif time.time()-startTime<duration:
                        control.updateData(joints)
                        control.getData()
                        control.ready4Img()
                        verts=control.kinect.vert
                        screen.fill((50,53,62))
                        BODY.update(key,verts)
                        BODY.render(cx,cy)
                        BODY.draw(screen)
                        pygame.display.flip()
                        key=pygame.key.get_pressed()
                    else:
                        return 
def main():
    pygame.init()
    screen=pygame.display.set_mode((800,600))
    generate(6,screen)
    screen.fill((255,255,255))
    #validate(screen)
    return

main()

