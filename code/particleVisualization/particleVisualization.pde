import hypermedia.net.*;

UDP udp;  // define the UDP object

//int cells = 128;
int cells = 156;


PVector[][] field = new PVector[cells][cells];
float t = random(3);
float opacity = 120;
int particleCount = 4000;

float colorVal = 0;
float speedVal = 3;

//TODO: Implement custom gradients (add some predefined ones too)

//TODO: Make an interface to calculate each flow cell's direction
//TODO: and/or magnitude

final PVector temp = new PVector();

Particle[] particles = new Particle[particleCount];

PVector getPosition(float ix, float iy) {
  return temp.set(ix/cells * width, iy/cells * height);
}

PVector getCell(float x, float y) {
  x = constrain(x, 0, width);
  y = constrain(y, 0, height);
  int ix = int(x/width * (field.length - 1));
  int iy = int(y/height * (field.length - 1));
  return field[ix][iy];
}

void setup() {
  size(1280, 720, P2D);
  //fullScreen();

  background(0);

  // create a new datagram connection on port 12000
  // and wait for incomming message
  udp = new UDP( this, 12000 );
  //udp.log( true );     // <-- printout the connection activity
  udp.listen( true );

  for (int i= 0; i<particles.length; i++) {
    particles[i] = new Particle(random(width), random(height));
  }

  //PVector pos = new PVector(width/2, height/2);
  //float maxD = sqrt(pow(width/2, 2) + pow(height/2, 2));
  for (int ix = 0; ix<cells; ix++) {
    for (int iy = 0; iy<cells; iy++) {
      field[ix][iy] = new PVector();
    }
  }
}

void draw() {
  noStroke();
  colorMode(RGB);
  fill(0, 16);
  rect(0, 0, width, height);
  colorMode(HSB, 360, 100, 100);
  //stroke(frameCount * 3 % 360, 100, 100, opacity);
  //// Change Color Here! int(colorVal)
  stroke(int(colorVal), 100, 100, opacity);    // first value mapping interval (0~120)
  strokeWeight(10);
  for (Particle p : particles) {
    p.update();
    p.draw();
  }
  for (int x = 0; x<cells; x++) {
    for (int y = 0; y<cells; y++) {
      PVector p = field[x][y];
      getPosition(x, y);
      float a;
      if (mousePressed) {
        a = atan2(mouseY - temp.y, mouseX - temp.x) - PI * 0.6;
      } else {
        a = noise(x/(float)cells*2, y/(float)cells*2, t + frameCount/100f) * TWO_PI * 1.5;
      }
      p.set(cos(a), sin(a));
    }
  }
}

class Particle {

  PVector pos, vel = new PVector(), acc = new PVector();
  PVector prev = new PVector();

  public Particle(float x, float y) {
    pos = new PVector(x, y);
  }

  private void resetPrev() {
    prev.set(pos.x, pos.y);
  }

  private void reset() {
    pos.set(random(width), random(height));
    resetPrev();
    vel.set(0, 0);
    acc.set(0, 0);
  }

  public void update() {
    resetPrev();
    if (pos.x < 0 || pos.x > width) {
      reset();
    }
    if (pos.y < 0 || pos.y > height) {
      reset();
    }
    acc = getCell(pos.x, pos.y);
    vel.add(acc);
    vel.limit(speedVal);    //This value 3~10
    pos.add(vel);
  }

  public void draw() {
    line(prev.x, prev.y, pos.x, pos.y);
  }
}

// UDP Recieve Module
void receive( byte[] data, String ip, int port ) {  // <-- extended handler
  
  // get the "real" message =
  // forget the ";\n" at the end <-- !!! only for a communication with Pd !!!
  data = subset(data, 0, data.length);
  String message = new String( data );
  
  colorVal = map(int(message), 0, 255, 120, 0);
  speedVal = map(int(message), 0, 255, 0.5, 5);
  
  // print the result
  //println( "receive: \""+message+"\" from "+ip+" on port "+port );
}
