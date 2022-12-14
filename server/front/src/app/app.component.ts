import {Component, OnInit, ViewChild} from '@angular/core';
import {WebcamImage, WebcamInitError} from "ngx-webcam";
import {forkJoin, Subject} from "rxjs";
import {MatStepper} from "@angular/material/stepper";
import {KerasService} from "./keras.service";
import {ImageVerification} from "./types";
import {DomSanitizer, SafeUrl} from "@angular/platform-browser";
const Speech = require('speak-tts').default;

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

  @ViewChild('stepper') stepper!: MatStepper;

  public captureTrigger = new Subject<void>();
  public image!: File | undefined;
  public imageURL!: String | undefined;
  public noisedImageURL!: SafeUrl | undefined;
  public denoisedImageURL!: SafeUrl | undefined;
  public speech: any;

  public robot_success = false;
  public robot_error = false;

  public verif!: ImageVerification | undefined;
  public caption!: string | undefined;

  constructor(
    private kerasService: KerasService,
    private sanitizer: DomSanitizer
  ) {}

  async ngOnInit() {
    this.speech = new Speech()
    await this.speech.init();
    await this.speech.speak({text: 'Hello there ! I am your Image Captioning assistant. Please give me a picture to analyse.', lang: 'en-US'})
  }

  public handleInitError(error: WebcamInitError): void {
    if (error.mediaStreamError && error.mediaStreamError.name === "NotAllowedError") {
      console.warn("Camera access was not allowed by user!");
    }
  }

  public async capture(image: WebcamImage): Promise<void> {
    this.imageURL = image.imageAsDataUrl;
    const response = await fetch(image.imageAsDataUrl);
    const buffer = await response.arrayBuffer();
    this.image = new File([buffer], 'image.jpg', {type: 'image/jpg'});
    this.robot_success = true;
  }

  public onFileSelected() {
    const reader = new FileReader();
    reader.onload = (_event) => this.imageURL = reader.result as string;
    this.image = (document.querySelector('#file') as any)?.files[0];
    if(this.image) reader.readAsDataURL(this.image);
    this.robot_success = true;
  }

  public verifyImage() {
    if(!this.image) return;
    this.resetRobot();
    // this.stepper.next();
    // this.speech.speak({text: 'Thank you. I am verifying your image.', lang: 'en-US'});
    this.kerasService.verify(this.image).subscribe(async verif => {
      this.verif = verif;
      if(verif.is_photo) {
        this.robot_success = true;
        await this.speech.speak({text: 'Your image is a valid photograph.', lang: 'en-US'});
      } else {
        this.robot_error = true;
        await this.speech.speak({text: 'Your image is not a photograph; please provide a valid photograph.', lang: 'en-US'});
      }
    });
  }

  public resetRobot() {
    this.robot_error = false;
    this.robot_success = false;
  }

  public reset(): void {
    this.image = undefined;
    this.verif = undefined;
    this.imageURL = undefined;
    this.caption = undefined;
    this.noisedImageURL = undefined;
    this.denoisedImageURL = undefined;
    this.stepper.reset();
    this.resetRobot();
  }

  public denoiseImage(): void {
    if(!this.image) return;
    this.resetRobot();
    // this.stepper.next();
    this.speech.speak({text: 'Let me reduce the noise on your image.', lang: 'en-US'});
    forkJoin([
      this.kerasService.noise(this.image),
      this.kerasService.denoise(this.image)
    ]).subscribe(async results => {
      this.robot_success = true;
      // this.speech.speak({text: 'The picture is no longer noised !', lang: 'en-US'});
      this.noisedImageURL = this.sanitizer.bypassSecurityTrustUrl(window.URL.createObjectURL(results[0]));
      this.denoisedImageURL = this.sanitizer.bypassSecurityTrustUrl(window.URL.createObjectURL(results[1]));
    });
  }

  public generateCaption() {
    if(!this.image) return;
    this.resetRobot();
    // this.stepper.next();
    // this.speech.speak({text: 'Let me take a look, it won\'t be too long...', lang: 'en-US'});
    this.kerasService.caption(this.image).subscribe(async caption => {
      this.caption = caption.caption;
      this.speech.speak({text: 'I think I understood: ' + caption.caption, lang: 'en-US'});
    });
  }

  public summary() {

  }
}
